# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import types 

from rsl_rl.modules import ActorCriticRMA
from rsl_rl.storage import RolloutStorage
import wandb
from rsl_rl.utils import unpad_trajectories
from rsl_rl.modules.actor_critic import PIEActorCritic

class RMS(object):
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape, device=device)
        self.S = torch.ones(shape, device=device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs + (delta**2) * self.n * bs / (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S

class PPO:
    actor_critic: ActorCriticRMA
    def __init__(self,
                 actor_critic,
                 estimator,
                 estimator_paras,
                 depth_encoder,
                 depth_encoder_paras,
                 depth_actor,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 dagger_update_freq=20,
                 priv_reg_coef_schedual = [0, 0, 0],
                 **kwargs
                 ):

        
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Adaptation
        self.hist_encoder_optimizer = optim.Adam(self.actor_critic.actor.history_encoder.parameters(), lr=learning_rate)
        self.priv_reg_coef_schedual = priv_reg_coef_schedual
        self.counter = 0

        # Estimator
        self.estimator = estimator
        self.priv_states_dim = estimator_paras["priv_states_dim"]
        self.num_prop = estimator_paras["num_prop"]
        self.num_scan = estimator_paras["num_scan"]
        self.estimator_optimizer = optim.Adam(self.estimator.parameters(), lr=estimator_paras["learning_rate"])
        self.train_with_estimated_states = estimator_paras["train_with_estimated_states"]

        # Depth encoder
        self.if_depth = depth_encoder is not None
        if self.if_depth:
            self.depth_encoder = depth_encoder
            self.depth_encoder_optimizer = optim.Adam(self.depth_encoder.parameters(), lr=depth_encoder_paras["learning_rate"])
            self.depth_encoder_paras = depth_encoder_paras
            self.depth_actor = depth_actor
            self.depth_actor_optimizer = optim.Adam([*self.depth_actor.parameters(), *self.depth_encoder.parameters()], lr=depth_encoder_paras["learning_rate"])

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, info, hist_encoding=False):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values, use proprio to compute estimated priv_states then actions, but store true priv_states
        if self.train_with_estimated_states:
            obs_est = obs.clone()
            priv_states_estimated = self.estimator(obs_est[:, :self.num_prop])
            obs_est[:, self.num_prop+self.num_scan:self.num_prop+self.num_scan+self.priv_states_dim] = priv_states_estimated
            self.transition.actions = self.actor_critic.act(obs_est, hist_encoding).detach()
        else:
            self.transition.actions = self.actor_critic.act(obs, hist_encoding).detach()

        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs

        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        rewards_total = rewards.clone()

        self.transition.rewards = rewards_total.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

        return rewards_total
    
    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
    
    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimator_loss = 0
        mean_discriminator_loss = 0
        mean_discriminator_acc = 0
        mean_priv_reg_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]) # match distribution dimension

                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy
                
                # Adaptation module update
                priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
                with torch.inference_mode():
                    hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
                priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
                priv_reg_stage = min(max((self.counter - self.priv_reg_coef_schedual[2]), 0) / self.priv_reg_coef_schedual[3], 1)
                priv_reg_coef = priv_reg_stage * (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0]) + self.priv_reg_coef_schedual[0]

                # Estimator
                priv_states_predicted = self.estimator(obs_batch[:, :self.num_prop])  # obs in batch is with true priv_states
                estimator_loss = (priv_states_predicted - obs_batch[:, self.num_prop+self.num_scan:self.num_prop+self.num_scan+self.priv_states_dim]).pow(2).mean()
                self.estimator_optimizer.zero_grad()
                estimator_loss.backward()
                nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
                self.estimator_optimizer.step()
                
                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + \
                       self.value_loss_coef * value_loss - \
                       self.entropy_coef * entropy_batch.mean() + \
                       priv_reg_coef * priv_reg_loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_estimator_loss += estimator_loss.item()
                mean_priv_reg_loss += priv_reg_loss.item()
                mean_discriminator_loss += 0
                mean_discriminator_acc += 0

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimator_loss /= num_updates
        mean_priv_reg_loss /= num_updates
        mean_discriminator_loss /= num_updates
        mean_discriminator_acc /= num_updates
        self.storage.clear()
        self.update_counter()
        return mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_discriminator_loss, mean_discriminator_acc, mean_priv_reg_loss, priv_reg_coef

    def update_dagger(self):
        mean_hist_latent_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
                with torch.inference_mode():
                    self.actor_critic.act(obs_batch, hist_encoding=True, masks=masks_batch, hidden_states=hid_states_batch[0])

                # Adaptation module update
                with torch.inference_mode():
                    priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
                hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
                hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
                self.hist_encoder_optimizer.zero_grad()
                hist_latent_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.actor.history_encoder.parameters(), self.max_grad_norm)
                self.hist_encoder_optimizer.step()
                
                mean_hist_latent_loss += hist_latent_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates
        self.storage.clear()
        self.update_counter()
        return mean_hist_latent_loss

    def update_depth_encoder(self, depth_latent_batch, scandots_latent_batch):
        # Depth encoder distillation
        if self.if_depth:
            depth_encoder_loss = (scandots_latent_batch.detach() - depth_latent_batch).norm(p=2, dim=1).mean()

            self.depth_encoder_optimizer.zero_grad()
            depth_encoder_loss.backward()
            nn.utils.clip_grad_norm_(self.depth_encoder.parameters(), self.max_grad_norm)
            self.depth_encoder_optimizer.step()
            return depth_encoder_loss.item()
        return 0.0
    
    def update_depth_actor(self, actions_student_batch, actions_teacher_batch, yaw_student_batch, yaw_teacher_batch):
        if self.if_depth:
            depth_actor_loss = (actions_teacher_batch.detach() - actions_student_batch).norm(p=2, dim=1).mean()
            yaw_loss = (yaw_teacher_batch.detach() - yaw_student_batch).norm(p=2, dim=1).mean()

            loss = depth_actor_loss + yaw_loss

            self.depth_actor_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.depth_actor.parameters(), self.max_grad_norm)
            self.depth_actor_optimizer.step()
            return depth_actor_loss.item(), yaw_loss.item()
        return 0.0, 0.0
    
    def update_depth_both(self, depth_latent_batch, scandots_latent_batch, actions_student_batch, actions_teacher_batch):
        if self.if_depth:
            depth_encoder_loss = (scandots_latent_batch.detach() - depth_latent_batch).norm(p=2, dim=1).mean()
            depth_actor_loss = (actions_teacher_batch.detach() - actions_student_batch).norm(p=2, dim=1).mean()

            depth_loss = depth_encoder_loss + depth_actor_loss

            self.depth_actor_optimizer.zero_grad()
            depth_loss.backward()
            nn.utils.clip_grad_norm_([*self.depth_actor.parameters(), *self.depth_encoder.parameters()], self.max_grad_norm)
            self.depth_actor_optimizer.step()
            return depth_encoder_loss.item(), depth_actor_loss.item()
        return 0.0, 0.0
    
    def update_counter(self):
        self.counter += 1


class PIEPPO:
    actor_critic: PIEActorCritic
    
    class PIETransition:
        """Custom transition class for PIE algorithm."""
        
        def __init__(self):
            # Standard PPO fields
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.values = None
            
            # PIE-specific fields
            self.depth_images = None
            self.prop_history = None
            self.base_velocity = None
            self.foot_clearance = None
            self.height_map_encoding = None
            self.latent_vector = None
            
            # Ground truth fields for estimator training
            self.true_velocity = None
            self.true_foot_clearance = None
            self.true_height_map = None
            self.true_next_state = None
        
        def clear(self):
            """Reset all fields to None."""
            # Standard PPO fields
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.values = None
            
            # PIE-specific fields
            self.depth_images = None
            self.prop_history = None
            self.base_velocity = None
            self.foot_clearance = None
            self.height_map_encoding = None
            self.latent_vector = None
            
            # Ground truth fields for estimator training
            self.true_velocity = None
            self.true_foot_clearance = None
            self.true_height_map = None
            self.true_next_state = None
        
        def __repr__(self):
            """String representation for debugging."""
            fields = []
            if self.observations is not None:
                fields.append(f"observations: tensor({self.observations.shape})")
            if self.critic_observations is not None:
                fields.append(f"critic_observations: tensor({self.critic_observations.shape})")
            if self.actions is not None:
                fields.append(f"actions: tensor({self.actions.shape})")
            if self.depth_images is not None:
                fields.append(f"depth_images: tensor({self.depth_images.shape})")
            if self.prop_history is not None:
                fields.append(f"prop_history: tensor({self.prop_history.shape})")
            
            return f"PIETransition({', '.join(fields)})"

    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 # PIE-specific parameters
                 velocity_loss_coef=1.0,
                 foot_clearance_loss_coef=1.0,
                 height_map_loss_coef=1.0,
                 kl_loss_coef=0.1,
                 next_state_loss_coef=1.0,
                 **kwargs):
        
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  
        
        self.optimizer = optim.Adam([
            {'params': self.actor_critic.actor.parameters()},
            {'params': self.actor_critic.critic.parameters()},
            {'params': self.actor_critic.estimator.parameters()}
        ], lr=learning_rate)
        
        # Use our nested PIETransition class
        self.transition = self.PIETransition()
        
        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # PIE-specific parameters
        self.velocity_loss_coef = velocity_loss_coef
        self.foot_clearance_loss_coef = foot_clearance_loss_coef
        self.height_map_loss_coef = height_map_loss_coef
        self.kl_loss_coef = kl_loss_coef
        self.next_state_loss_coef = next_state_loss_coef

    def init_storage(self, num_envs, num_transitions_per_env, obs_shape, critic_obs_shape, action_shape):
        """Initialize storage with PIE-specific buffers."""
        print(f"Initializing PIE storage with shapes: obs={obs_shape}, critic_obs={critic_obs_shape}, action={action_shape}")
        
        try:
            # Make sure we're using the correct observation shape - only proprioceptive states
            num_proprio = getattr(self.actor_critic, 'num_proprio', obs_shape[0])
            
            # If obs_shape is larger than num_proprio, we should use num_proprio instead
            if obs_shape[0] > num_proprio:
                print(f"WARNING: Observation shape {obs_shape[0]} is larger than proprioceptive size {num_proprio}")
                print(f"Using only the proprioceptive part for storage")
                obs_shape = [num_proprio]
            
            # First create basic storage
            from rsl_rl.storage import RolloutStorage
            self.storage = RolloutStorage(
                num_envs, 
                num_transitions_per_env, 
                obs_shape,  
                critic_obs_shape, 
                action_shape, 
                self.device
            )
            
            # Initialize PIE-specific buffers from actor_critic parameters
            velocity_dim = getattr(self.actor_critic, 'velocity_dim', 3)
            foot_clearance_dim = getattr(self.actor_critic, 'foot_clearance_dim', 4)
            height_map_dim = getattr(self.actor_critic, 'height_map_dim', 32)
            latent_dim = getattr(self.actor_critic, 'latent_dim', 32)
            
            # Get dimensions for depth images - provide a default if not available
            if hasattr(self.actor_critic, 'depth_shape'):
                depth_shape = self.actor_critic.depth_shape
            else:
                # Use smaller default size to save memory
                depth_shape = (28, 28)  # Reduced size from paper to save memory
                print(f"Using reduced depth image shape: {depth_shape}")
            
            # Get history dimensions
            hist_len = getattr(self.actor_critic, 'hist_len', 10)
            prop_history_shape = (hist_len, num_proprio)
            
            # Add our memory-efficient extensions
            print("Adding memory-efficient PIE buffer extensions to RolloutStorage")
            self._init_pie_buffers(
                num_envs, 
                num_transitions_per_env, 
                depth_shape, 
                prop_history_shape,
                velocity_dim, 
                foot_clearance_dim, 
                height_map_dim, 
                latent_dim
            )
            print("Memory-efficient PIE buffers initialized successfully")
            
        except Exception as e:
            print(f"Error initializing PIE buffers: {e}")
            # At minimum, make sure basic storage is initialized
            if self.storage is None:
                from rsl_rl.storage import RolloutStorage
                self.storage = RolloutStorage(
                    num_envs, 
                    num_transitions_per_env, 
                    obs_shape, 
                    critic_obs_shape, 
                    action_shape, 
                    self.device
                )

    def _init_pie_buffers(self, num_envs, num_transitions_per_env, depth_shape, prop_history_shape, 
                        velocity_dim, foot_clearance_dim, height_map_dim, latent_dim):
        """Memory-efficient implementation of storage extensions to avoid GPU OOM errors."""
        import types
        
        # Initialize buffers on the storage object - using CPU for larger tensors
        self.storage.depth_images = torch.zeros(
            num_transitions_per_env, 
            num_envs, 
            2,  # Assuming 2-channel depth
            depth_shape[0], 
            depth_shape[1], 
            device="cpu"  # Store on CPU to save GPU memory
        )
        
        self.storage.prop_history = torch.zeros(
            num_transitions_per_env,
            num_envs,
            prop_history_shape[0],  # hist_len
            prop_history_shape[1],  # num_proprio
            dtype=torch.float16,  # Use half precision to save memory
            device=self.device
        )
        
        # For storing estimated values
        self.storage.base_velocity = torch.zeros(
            num_transitions_per_env,
            num_envs, 
            velocity_dim,
            device=self.device
        )
        
        self.storage.foot_clearance = torch.zeros(
            num_transitions_per_env,
            num_envs, 
            foot_clearance_dim,
            device=self.device
        )
        
        self.storage.height_map_encoding = torch.zeros(
            num_transitions_per_env,
            num_envs, 
            height_map_dim,
            device=self.device
        )
        
        self.storage.latent_vector = torch.zeros(
            num_transitions_per_env,
            num_envs, 
            latent_dim,
            device=self.device
        )
        
        # Initialize ground truth value buffers to None (will create when needed)
        self.storage.true_velocity = None
        self.storage.true_foot_clearance = None
        self.storage.true_height_map = None
        self.storage.true_next_state = None
        
        # Add methods to the storage class
        def get_depth_images_batch(storage_self, indices=None):
            if not hasattr(storage_self, 'depth_images') or storage_self.depth_images is None:
                return None
                
            if indices is None:
                # Get all data
                depth_batch = storage_self.depth_images.view(-1, *storage_self.depth_images.shape[2:])
            else:
                # Get specific indices
                depth_batch = storage_self.depth_images.view(-1, *storage_self.depth_images.shape[2:])[indices]
            # Move to GPU for processing
            return depth_batch.to(storage_self.device)
        
        def get_prop_history_batch(storage_self, indices=None):
            if not hasattr(storage_self, 'prop_history') or storage_self.prop_history is None:
                return None
                
            if indices is None:
                prop_batch = storage_self.prop_history.view(-1, *storage_self.prop_history.shape[2:])
            else:
                prop_batch = storage_self.prop_history.view(-1, *storage_self.prop_history.shape[2:])[indices]
            # Convert back to full precision
            return prop_batch.float()
        
        def get_true_velocity_batch(storage_self, indices=None):
            if not hasattr(storage_self, 'true_velocity') or storage_self.true_velocity is None:
                return None
            if indices is None:
                return storage_self.true_velocity.view(-1, storage_self.true_velocity.shape[-1])
            else:
                return storage_self.true_velocity.view(-1, storage_self.true_velocity.shape[-1])[indices]
        
        def get_true_foot_clearance_batch(storage_self, indices=None):
            if not hasattr(storage_self, 'true_foot_clearance') or storage_self.true_foot_clearance is None:
                return None
            if indices is None:
                return storage_self.true_foot_clearance.view(-1, storage_self.true_foot_clearance.shape[-1])
            else:
                return storage_self.true_foot_clearance.view(-1, storage_self.true_foot_clearance.shape[-1])[indices]
        
        def get_true_height_map_batch(storage_self, indices=None):
            if not hasattr(storage_self, 'true_height_map') or storage_self.true_height_map is None:
                return None
            if indices is None:
                return storage_self.true_height_map.view(-1, storage_self.true_height_map.shape[-1])
            else:
                return storage_self.true_height_map.view(-1, storage_self.true_height_map.shape[-1])[indices]
        
        def get_true_next_state_batch(storage_self, indices=None):
            if not hasattr(storage_self, 'true_next_state') or storage_self.true_next_state is None:
                return None
            if indices is None:
                return storage_self.true_next_state.view(-1, storage_self.true_next_state.shape[-1])
            else:
                return storage_self.true_next_state.view(-1, storage_self.true_next_state.shape[-1])[indices]
        
        # Define add_transitions method for storage
        def pie_add_transitions(storage_self, transition):
            """Memory-efficient version of add_transitions that handles None values."""
            if storage_self.step >= storage_self.num_transitions_per_env:
                storage_self.step = 0
            
            # Validate inputs before copying to prevent null pointer errors
            if not hasattr(transition, 'observations') or transition.observations is None:
                raise ValueError("transition.observations cannot be None")
            
            # Store standard PPO values
            storage_self.observations[storage_self.step].copy_(transition.observations)
            
            # Handle critic observations - use observations as fallback
            if hasattr(transition, 'critic_observations') and transition.critic_observations is not None:
                if hasattr(storage_self, 'privileged_observations') and storage_self.privileged_observations is not None:
                    storage_self.privileged_observations[storage_self.step].copy_(transition.critic_observations)
                elif hasattr(transition, 'critic_observations') and transition.critic_observations is not None:
                    # Initialize privileged_observations if needed
                    storage_self.privileged_observations = torch.zeros(
                        storage_self.num_transitions_per_env,
                        storage_self.num_envs,
                        *transition.critic_observations.shape[1:],
                        device=storage_self.device
                    )
                    storage_self.privileged_observations[storage_self.step].copy_(transition.critic_observations)
            
            # Validate required fields
            if not hasattr(transition, 'actions') or transition.actions is None:
                raise ValueError("transition.actions cannot be None")
            if not hasattr(transition, 'rewards') or transition.rewards is None:
                raise ValueError("transition.rewards cannot be None")
            if not hasattr(transition, 'dones') or transition.dones is None:
                raise ValueError("transition.dones cannot be None")
            if not hasattr(transition, 'actions_log_prob') or transition.actions_log_prob is None:
                raise ValueError("transition.actions_log_prob cannot be None")
            if not hasattr(transition, 'action_mean') or transition.action_mean is None:
                raise ValueError("transition.action_mean cannot be None")
            if not hasattr(transition, 'action_sigma') or transition.action_sigma is None:
                raise ValueError("transition.action_sigma cannot be None")
            
            # Copy required fields
            storage_self.actions[storage_self.step].copy_(transition.actions)
            storage_self.rewards[storage_self.step].copy_(transition.rewards.view(-1, 1))
            storage_self.dones[storage_self.step].copy_(transition.dones.view(-1, 1))
            storage_self.actions_log_prob[storage_self.step].copy_(transition.actions_log_prob.view(-1, 1))
            storage_self.mu[storage_self.step].copy_(transition.action_mean)
            storage_self.sigma[storage_self.step].copy_(transition.action_sigma)
            
            # Store PIE-specific data if available - with thorough validation
            if hasattr(storage_self, 'depth_images') and hasattr(transition, 'depth_images') and transition.depth_images is not None:
                # Move to CPU to save memory
                cpu_depth = transition.depth_images if transition.depth_images.device.type == 'cpu' else transition.depth_images.cpu()
                storage_self.depth_images[storage_self.step].copy_(cpu_depth)
            
            if hasattr(storage_self, 'prop_history') and hasattr(transition, 'prop_history') and transition.prop_history is not None:
                # Convert to half precision
                half_prop = transition.prop_history.half() if transition.prop_history.dtype != torch.float16 else transition.prop_history
                storage_self.prop_history[storage_self.step].copy_(half_prop)
            
            if hasattr(storage_self, 'base_velocity') and hasattr(transition, 'base_velocity') and transition.base_velocity is not None:
                storage_self.base_velocity[storage_self.step].copy_(transition.base_velocity)
            
            if hasattr(storage_self, 'foot_clearance') and hasattr(transition, 'foot_clearance') and transition.foot_clearance is not None:
                storage_self.foot_clearance[storage_self.step].copy_(transition.foot_clearance)
            
            if hasattr(storage_self, 'height_map_encoding') and hasattr(transition, 'height_map_encoding') and transition.height_map_encoding is not None:
                storage_self.height_map_encoding[storage_self.step].copy_(transition.height_map_encoding)
            
            if hasattr(storage_self, 'latent_vector') and hasattr(transition, 'latent_vector') and transition.latent_vector is not None:
                storage_self.latent_vector[storage_self.step].copy_(transition.latent_vector)
            
            # Handle ground truth values if available
            if hasattr(storage_self, 'true_velocity') and hasattr(transition, 'true_velocity') and transition.true_velocity is not None:
                if storage_self.true_velocity is None:
                    # Initialize storage for true_velocity if not done yet
                    velocity_dim = transition.true_velocity.shape[-1]
                    storage_self.true_velocity = torch.zeros(
                        storage_self.num_transitions_per_env, 
                        storage_self.num_envs, 
                        velocity_dim,
                        device=storage_self.device
                    )
                storage_self.true_velocity[storage_self.step].copy_(transition.true_velocity)
            
            if hasattr(storage_self, 'true_foot_clearance') and hasattr(transition, 'true_foot_clearance') and transition.true_foot_clearance is not None:
                if storage_self.true_foot_clearance is None:
                    foot_clearance_dim = transition.true_foot_clearance.shape[-1]
                    storage_self.true_foot_clearance = torch.zeros(
                        storage_self.num_transitions_per_env,
                        storage_self.num_envs, 
                        foot_clearance_dim,
                        device=storage_self.device
                    )
                storage_self.true_foot_clearance[storage_self.step].copy_(transition.true_foot_clearance)
            
            if hasattr(storage_self, 'true_height_map') and hasattr(transition, 'true_height_map') and transition.true_height_map is not None:
                if storage_self.true_height_map is None:
                    height_map_dim = transition.true_height_map.shape[-1]
                    storage_self.true_height_map = torch.zeros(
                        storage_self.num_transitions_per_env,
                        storage_self.num_envs, 
                        height_map_dim,
                        device=storage_self.device
                    )
                storage_self.true_height_map[storage_self.step].copy_(transition.true_height_map)
            
            if hasattr(storage_self, 'true_next_state') and hasattr(transition, 'true_next_state') and transition.true_next_state is not None:
                if storage_self.true_next_state is None:
                    state_dim = transition.true_next_state.shape[-1]
                    storage_self.true_next_state = torch.zeros(
                        storage_self.num_transitions_per_env,
                        storage_self.num_envs, 
                        state_dim,
                        device=storage_self.device
                    )
                storage_self.true_next_state[storage_self.step].copy_(transition.true_next_state)
            
            # Increment step
            storage_self.step += 1
        
        # Bind the methods to the storage instance
        self.storage.get_depth_images_batch = types.MethodType(get_depth_images_batch, self.storage)
        self.storage.get_prop_history_batch = types.MethodType(get_prop_history_batch, self.storage)
        self.storage.get_true_velocity_batch = types.MethodType(get_true_velocity_batch, self.storage)
        self.storage.get_true_foot_clearance_batch = types.MethodType(get_true_foot_clearance_batch, self.storage)
        self.storage.get_true_height_map_batch = types.MethodType(get_true_height_map_batch, self.storage)
        self.storage.get_true_next_state_batch = types.MethodType(get_true_next_state_batch, self.storage)
        
        # Bind our add_transitions function to the storage object
        self.storage.add_transitions = types.MethodType(pie_add_transitions, self.storage)

    def act(self, proprio_obs, depth_images, prop_history, info=None):
        """Get actions from the actor critic based on proprio observations, depth images and history."""
        print(f"ACT CALLED: proprio_obs shape = {proprio_obs.shape if proprio_obs is not None else None}")
        
        # Clear any existing data in the transition
        self.transition.clear()
        
        # Make sure we're dealing with tensors
        if proprio_obs is None:
            raise ValueError("proprio_obs cannot be None")
        
        # Run the PIE estimator and actor
        actions, base_velocity, foot_clearance, height_map_encoding, latent_vector = self.actor_critic.act(
            proprio_obs, depth_images, prop_history, return_estimations=True
        )
        
        print(f"  - Got actions shape: {actions.shape if actions is not None else None}")
        print(f"  - Got base_velocity shape: {base_velocity.shape if base_velocity is not None else None}")
        
        # Initialize the transition object with observations
        # Just use proprio_obs directly without attempting to pad it
        self.transition.observations = proprio_obs.clone()
        
        # For critic observations, use privileged info if available, otherwise use proprio
        if info is not None and "critic_observations" in info and info["critic_observations"] is not None:
            self.transition.critic_observations = info["critic_observations"]
        else:
            self.transition.critic_observations = proprio_obs.clone()
        
        # Store depth and history data 
        self.transition.depth_images = depth_images
        self.transition.prop_history = prop_history
        
        # Store action data
        self.transition.actions = actions.detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        
        # Get value estimate
        self.transition.values = self.actor_critic.evaluate(self.transition.critic_observations).detach()
        
        # Store PIE-specific estimations
        self.transition.base_velocity = base_velocity.detach()
        self.transition.foot_clearance = foot_clearance.detach()
        self.transition.height_map_encoding = height_map_encoding.detach()
        self.transition.latent_vector = latent_vector.detach()
        
        # Verify that all required fields are set
        print(f"  - Set transition.observations: {self.transition.observations is not None}, shape: {self.transition.observations.shape}")
        print(f"  - Set transition.critic_observations: {self.transition.critic_observations is not None}, shape: {self.transition.critic_observations.shape if self.transition.critic_observations is not None else None}")
        print(f"  - Set transition.actions: {self.transition.actions is not None}, shape: {self.transition.actions.shape if self.transition.actions is not None else None}")
        
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        """Process environment step and add to storage."""
        if self.storage is None:
            raise RuntimeError("Storage has not been initialized. Call init_storage() before training.")
            
        rewards_total = rewards.clone()
        
        self.transition.rewards = rewards_total.clone()
        self.transition.dones = dones
        
        # Bootstrapping on time outs
        if isinstance(infos, dict) and 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1
            )

        # Store actual ground truth values if available (for training the estimator)
        # Only set these if the values are not None
        if isinstance(infos, dict):
            if 'true_velocity' in infos and infos['true_velocity'] is not None:
                self.transition.true_velocity = infos['true_velocity'].to(self.device)
            
            if 'true_foot_clearance' in infos and infos['true_foot_clearance'] is not None:
                self.transition.true_foot_clearance = infos['true_foot_clearance'].to(self.device)
            
            if 'true_height_map' in infos and infos['true_height_map'] is not None:
                self.transition.true_height_map = infos['true_height_map'].to(self.device)
            
            if 'next_state' in infos and infos['next_state'] is not None:
                self.transition.true_next_state = infos['next_state'].to(self.device)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        
        return rewards_total
    
    def compute_returns(self, last_critic_obs):
        """Compute returns for advantage estimation."""
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
    
    def update(self):
        """Update policy and estimator using PPO algorithm with more careful handling of gradients."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_velocity_loss = 0
        mean_foot_clearance_loss = 0
        mean_height_map_loss = 0
        mean_kl_loss = 0
        mean_next_state_loss = 0
        
        # Determine the right generator based on whether we're using a recurrent policy
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        # Process each mini-batch
        num_updates = 0
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
            old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
            
            num_updates += 1
            
            # ==================== CRITICALLY IMPORTANT ====================
            # Detach all tensors to ensure there are no connections to previous graphs
            obs_batch = obs_batch.detach()
            critic_obs_batch = critic_obs_batch.detach()
            actions_batch = actions_batch.detach()
            target_values_batch = target_values_batch.detach()
            advantages_batch = advantages_batch.detach()
            returns_batch = returns_batch.detach()
            old_actions_log_prob_batch = old_actions_log_prob_batch.detach()
            old_mu_batch = old_mu_batch.detach()
            old_sigma_batch = old_sigma_batch.detach()
            
            # Always zero gradients at the beginning of each mini-batch processing
            self.optimizer.zero_grad()
            
            # Get current batch size
            current_batch_size = obs_batch.size(0)
            
            # Extract proprioceptive part for the PIE estimator
            proprio_batch = obs_batch[:, :self.actor_critic.num_proprio]
            
            # Get depth and prop history data for this mini-batch - with detaching
            try:
                depth_batch = self.storage.get_depth_images_batch(None)
                if depth_batch is not None:
                    depth_batch = depth_batch[:current_batch_size].detach()
            except:
                depth_batch = None
                
            try:
                prop_history_batch = self.storage.get_prop_history_batch(None)
                if prop_history_batch is not None:
                    prop_history_batch = prop_history_batch[:current_batch_size].detach()
            except:
                prop_history_batch = None
            
            # Get ground truth data and ensure they're fully detached from any graphs
            try:
                true_velocity_batch = self.storage.get_true_velocity_batch(None)
                if true_velocity_batch is not None:
                    true_velocity_batch = true_velocity_batch[:current_batch_size].detach()
            except:
                true_velocity_batch = None
                
            try:
                true_foot_clearance_batch = self.storage.get_true_foot_clearance_batch(None)
                if true_foot_clearance_batch is not None:
                    true_foot_clearance_batch = true_foot_clearance_batch[:current_batch_size].detach()
            except:
                true_foot_clearance_batch = None
                
            try:
                true_height_map_batch = self.storage.get_true_height_map_batch(None)
                if true_height_map_batch is not None:
                    true_height_map_batch = true_height_map_batch[:current_batch_size].detach()
            except:
                true_height_map_batch = None
                
            try:
                true_next_state_batch = self.storage.get_true_next_state_batch(None)
                if true_next_state_batch is not None:
                    true_next_state_batch = true_next_state_batch[:current_batch_size].detach()
            except:
                true_next_state_batch = None

            # Run estimator forward pass with error handling
            try:
                base_velocity, foot_clearance, height_map_encoding, latent_vector, latent_mu, latent_logvar, next_state_pred = \
                    self.actor_critic.estimator(depth_batch, prop_history_batch)
            except Exception as e:
                print(f"Error in estimator forward pass: {e}")
                # Create dummy tensors to continue training if possible
                batch_size = proprio_batch.size(0)
                base_velocity = torch.zeros(batch_size, 3, device=self.device)
                foot_clearance = torch.zeros(batch_size, 4, device=self.device)
                height_map_encoding = torch.zeros(batch_size, 32, device=self.device)
                latent_vector = torch.zeros(batch_size, 32, device=self.device)
                latent_mu = torch.zeros(batch_size, 32, device=self.device)
                latent_logvar = torch.zeros(batch_size, 32, device=self.device)
                next_state_pred = torch.zeros(batch_size, proprio_batch.size(1), device=self.device)

            # Run actor forward pass
            try:
                dummy_actions = self.actor_critic.actor(
                    proprio_batch, 
                    base_velocity, 
                    foot_clearance, 
                    height_map_encoding, 
                    latent_vector
                )
                self.actor_critic.distribution = torch.distributions.Normal(dummy_actions, self.actor_critic.std)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            except Exception as e:
                print(f"Error in actor forward pass: {e}")
                # Create dummy tensors
                dummy_actions = torch.zeros_like(actions_batch)
                actions_log_prob_batch = torch.zeros(actions_batch.size(0), 1, device=self.device)
            
            # Run critic forward pass
            try:
                value_batch = self.actor_critic.evaluate(critic_obs_batch)
                entropy_batch = self.actor_critic.entropy
            except Exception as e:
                print(f"Error in critic forward pass: {e}")
                # Create dummy tensors
                value_batch = torch.zeros_like(target_values_batch)
                entropy_batch = torch.zeros(actions_batch.size(0), device=self.device)

            # Initialize our losses to zero tensors
            surrogate_loss = torch.tensor(0.0, device=self.device)
            value_loss = torch.tensor(0.0, device=self.device)
            velocity_loss = torch.tensor(0.0, device=self.device)
            foot_clearance_loss = torch.tensor(0.0, device=self.device)
            height_map_loss = torch.tensor(0.0, device=self.device)
            next_state_loss = torch.tensor(0.0, device=self.device)
            kl_loss = torch.tensor(0.0, device=self.device)
            
            # Calculate PPO surrogate loss with error handling
            try:
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            except Exception as e:
                print(f"Error in surrogate loss calculation: {e}")

            # Calculate value function loss with error handling
            try:
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
            except Exception as e:
                print(f"Error in value loss calculation: {e}")
            
            # Calculate estimator losses with error handling
            try:
                if true_velocity_batch is not None:
                    velocity_loss = F.mse_loss(base_velocity, true_velocity_batch)
            except Exception as e:
                print(f"Error in velocity loss calculation: {e}")
            
            try:
                if true_foot_clearance_batch is not None:
                    foot_clearance_loss = F.mse_loss(foot_clearance, true_foot_clearance_batch)
            except Exception as e:
                print(f"Error in foot clearance loss calculation: {e}")
            
            try:
                if true_height_map_batch is not None:
                    height_map_loss = F.mse_loss(height_map_encoding, true_height_map_batch)
            except Exception as e:
                print(f"Error in height map loss calculation: {e}")
            
            try:
                if true_next_state_batch is not None:
                    next_state_loss = F.mse_loss(next_state_pred, true_next_state_batch)
            except Exception as e:
                print(f"Error in next state loss calculation: {e}")
            
            # Calculate KL loss for VAE with error handling
            try:
                kl_loss = -0.5 * torch.sum(1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp())
                kl_loss = kl_loss / latent_mu.size(0)  # Normalize by batch size
            except Exception as e:
                print(f"Error in KL loss calculation: {e}")
            
            # Combine all losses
            try:
                loss = surrogate_loss + \
                    self.value_loss_coef * value_loss - \
                    self.entropy_coef * entropy_batch.mean() + \
                    self.velocity_loss_coef * velocity_loss + \
                    self.foot_clearance_loss_coef * foot_clearance_loss + \
                    self.height_map_loss_coef * height_map_loss + \
                    self.kl_loss_coef * kl_loss + \
                    self.next_state_loss_coef * next_state_loss
                    
                # *** KEY CHANGE: Use retain_graph=True to avoid the backward error ***
                loss.backward(retain_graph=True)
                
                # Clip gradients
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                
                # Update parameters
                self.optimizer.step()
            except Exception as e:
                print(f"Error during loss combination or backward pass: {e}")
            
            # Accumulate losses for reporting - detach to avoid memory leaks
            mean_value_loss += value_loss.detach().item()
            mean_surrogate_loss += surrogate_loss.detach().item()
            mean_velocity_loss += velocity_loss.detach().item()
            mean_foot_clearance_loss += foot_clearance_loss.detach().item()
            mean_height_map_loss += height_map_loss.detach().item()
            mean_kl_loss += kl_loss.detach().item()
            mean_next_state_loss += next_state_loss.detach().item()
            
            # Force cleanup of tensors
            del obs_batch, critic_obs_batch, actions_batch, target_values_batch
            del advantages_batch, returns_batch, old_actions_log_prob_batch
            del old_mu_batch, old_sigma_batch, proprio_batch
            if depth_batch is not None: del depth_batch
            if prop_history_batch is not None: del prop_history_batch
            if true_velocity_batch is not None: del true_velocity_batch
            if true_foot_clearance_batch is not None: del true_foot_clearance_batch
            if true_height_map_batch is not None: del true_height_map_batch
            if true_next_state_batch is not None: del true_next_state_batch
            
            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate means based on actual number of updates
        if num_updates > 0:
            mean_value_loss /= num_updates
            mean_surrogate_loss /= num_updates
            mean_velocity_loss /= num_updates
            mean_foot_clearance_loss /= num_updates
            mean_height_map_loss /= num_updates
            mean_kl_loss /= num_updates
            mean_next_state_loss /= num_updates
        
        # Clear storage for next update - USING OUR CUSTOM CLEAR METHOD
        self.custom_clear_storage()
        
        # Return mean losses
        return {
            'value_loss': mean_value_loss,
            'surrogate_loss': mean_surrogate_loss,
            'velocity_loss': mean_velocity_loss,
            'foot_clearance_loss': mean_foot_clearance_loss,
            'height_map_loss': mean_height_map_loss,
            'kl_loss': mean_kl_loss,
            'next_state_loss': mean_next_state_loss,
            'learning_rate': self.learning_rate
        }

    def custom_clear_storage(self):
        """Custom method to clear storage without calling super().clear()"""
        # Reset step counter directly
        if hasattr(self.storage, 'step'):
            self.storage.step = 0
        
        # Clear PIE-specific attributes
        if hasattr(self.storage, 'true_velocity') and self.storage.true_velocity is not None:
            self.storage.true_velocity = None
        if hasattr(self.storage, 'true_foot_clearance') and self.storage.true_foot_clearance is not None:
            self.storage.true_foot_clearance = None
        if hasattr(self.storage, 'true_height_map') and self.storage.true_height_map is not None:
            self.storage.true_height_map = None
        if hasattr(self.storage, 'true_next_state') and self.storage.true_next_state is not None:
            self.storage.true_next_state = None
            
        # Force garbage collection after clearing storage
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()