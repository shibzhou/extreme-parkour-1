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

import numpy as np

import code
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU
import torch.nn.functional as F

from .pie_estimator import PIEEstimator



class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        # self.device = device
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10
        # last_activation = nn.ELU()

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), self.activation_fn
                )

    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        # print("obs device", obs.device)
        # print("encoder device", next(self.encoder.parameters()).device)
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output

class Actor(nn.Module):
    def __init__(self, num_prop, 
                 num_scan, 
                 num_actions, 
                 scan_encoder_dims,
                 actor_hidden_dims, 
                 priv_encoder_dims, 
                 num_priv_latent, 
                 num_priv_explicit, 
                 num_hist, activation, 
                 tanh_encoder_output=False) -> None:
        super().__init__()
        # prop -> scan -> priv_explicit -> priv_latent -> hist
        # actor input: prop -> scan -> priv_explicit -> latent
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.num_priv_explicit = num_priv_explicit
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0

        if len(priv_encoder_dims) > 0:
                    priv_encoder_layers = []
                    priv_encoder_layers.append(nn.Linear(num_priv_latent, priv_encoder_dims[0]))
                    priv_encoder_layers.append(activation)
                    for l in range(len(priv_encoder_dims) - 1):
                        priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
                        priv_encoder_layers.append(activation)
                    self.priv_encoder = nn.Sequential(*priv_encoder_layers)
                    priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, priv_encoder_output_dim)

        if self.if_scan_encode:
            scan_encoder = []
            scan_encoder.append(nn.Linear(num_scan, scan_encoder_dims[0]))
            scan_encoder.append(activation)
            for l in range(len(scan_encoder_dims) - 1):
                if l == len(scan_encoder_dims) - 2:
                    scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l+1]))
                    scan_encoder.append(nn.Tanh())
                else:
                    scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l + 1]))
                    scan_encoder.append(activation)
            self.scan_encoder = nn.Sequential(*scan_encoder)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan
        
        actor_layers = []
        actor_layers.append(nn.Linear(num_prop+
                                      self.scan_encoder_output_dim+
                                      num_priv_explicit+
                                      priv_encoder_output_dim, 
                                      actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        if tanh_encoder_output:
            actor_layers.append(nn.Tanh())
        self.actor_backbone = nn.Sequential(*actor_layers)

    def forward(self, obs, hist_encoding: bool, eval=False, scandots_latent=None):
        if not eval:
            if self.if_scan_encode:
                obs_scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
                if scandots_latent is None:
                    scan_latent = self.scan_encoder(obs_scan)   
                else:
                    scan_latent = scandots_latent
                obs_prop_scan = torch.cat([obs[:, :self.num_prop], scan_latent], dim=1)
            else:
                obs_prop_scan = obs[:, :self.num_prop + self.num_scan]
            obs_priv_explicit = obs[:, self.num_prop + self.num_scan:self.num_prop + self.num_scan + self.num_priv_explicit]
            if hist_encoding:
                latent = self.infer_hist_latent(obs)
            else:
                latent = self.infer_priv_latent(obs)
            backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)
            backbone_output = self.actor_backbone(backbone_input)
            return backbone_output
        else:
            if self.if_scan_encode:
                obs_scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
                if scandots_latent is None:
                    scan_latent = self.scan_encoder(obs_scan)   
                else:
                    scan_latent = scandots_latent
                obs_prop_scan = torch.cat([obs[:, :self.num_prop], scan_latent], dim=1)
            else:
                obs_prop_scan = obs[:, :self.num_prop + self.num_scan]
            obs_priv_explicit = obs[:, self.num_prop + self.num_scan:self.num_prop + self.num_scan + self.num_priv_explicit]
            if hist_encoding:
                latent = self.infer_hist_latent(obs)
            else:
                latent = self.infer_priv_latent(obs)
            backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)
            backbone_output = self.actor_backbone(backbone_input)
            return backbone_output
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan + self.num_priv_explicit: self.num_prop + self.num_scan + self.num_priv_explicit + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)

class ActorCriticRMA(nn.Module):
    is_recurrent = False
    def __init__(self,  num_prop,
                        num_scan,
                        num_critic_obs,
                        num_priv_latent, 
                        num_priv_explicit,
                        num_hist,
                        num_actions,
                        scan_encoder_dims=[256, 256, 256],
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticRMA, self).__init__()

        self.kwargs = kwargs
        priv_encoder_dims= kwargs['priv_encoder_dims']
        activation = get_activation(activation)
        
        self.actor = Actor(num_prop, num_scan, num_actions, scan_encoder_dims, actor_hidden_dims, priv_encoder_dims, num_priv_latent, num_priv_explicit, num_hist, activation, tanh_encoder_output=kwargs['tanh_encoder_output'])
        

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, hist_encoding):
        mean = self.actor(observations, hist_encoding)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, hist_encoding=False, **kwargs):
        self.update_distribution(observations, hist_encoding)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, hist_encoding=False, eval=False, scandots_latent=None, **kwargs):
        if not eval:
            actions_mean = self.actor(observations, hist_encoding, eval, scandots_latent)
            return actions_mean
        else:
            actions_mean, latent_hist, latent_priv = self.actor(observations, hist_encoding, eval=True)
            return actions_mean, latent_hist, latent_priv

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
    def reset_std(self, std, num_actions, device):
        new_std = std * torch.ones(num_actions, device=device)
        self.std.data = new_std.data

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

'''

proprio_obs: current proprioceptive observation (o_t)
base_velocity: current base velocity (v_t)
foot_clearance: current foot clearance (h^f_t)
height_map_encoding: encoding of the height map (z^m_t)
latent_vector: latent vector (z_t)

'''

class PIEActor(nn.Module):
    def __init__(self, num_proprio, 
                 num_actions, 
                 velocity_dim=3, 
                 foot_clearance_dim=4, 
                 height_map_dim=32, 
                 latent_dim=32,
                 hidden_dims=[512, 256, 128], 
                 activation='elu'):
        super(PIEActor, self).__init__()
        
        self.num_proprio = num_proprio
        activation_fn = get_activation(activation)
        
        total_input_dim = (num_proprio + 
                           velocity_dim + 
                           foot_clearance_dim + 
                           height_map_dim + 
                           latent_dim)
        
        actor_layers = []
        actor_layers.append(nn.Linear(total_input_dim, hidden_dims[0]))
        actor_layers.append(activation_fn)
        
        for l in range(len(hidden_dims) - 1):
            actor_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l+1]))
            actor_layers.append(activation_fn)
            
        actor_layers.append(nn.Linear(hidden_dims[-1], num_actions))
        
        self.actor = nn.Sequential(*actor_layers)
    
    def forward(self, proprio_obs, base_velocity, foot_clearance, height_map_encoding, latent_vector):
        inputs = torch.cat([
            proprio_obs,
            base_velocity,
            foot_clearance,
            height_map_encoding,
            latent_vector
        ], dim=1)
        
        actions = self.actor(inputs)
        
        return actions

class PIEActorCritic(nn.Module):
    is_recurrent = False  # Add this attribute so PPO.update() can check it
    
    def __init__(self,
                num_proprio,
                num_scan,
                num_critic_obs, 
                velocity_dim=3,
                foot_clearance_dim=4,
                height_map_dim=32,
                latent_dim=32,
                hist_len=10,
                num_actions=12,  # Default to 12 for quadruped robots (3 joints × 4 legs)
                actor_hidden_dims=[512, 256, 128],
                critic_hidden_dims=[512, 256, 128],
                activation='elu',
                init_noise_std=1.0):
        super(PIEActorCritic, self).__init__()

        # Store dimensions as class attributes for later use
        self.num_proprio = num_proprio
        self.velocity_dim = velocity_dim
        self.foot_clearance_dim = foot_clearance_dim
        self.height_map_dim = height_map_dim
        self.latent_dim = latent_dim
        self.hist_len = hist_len
        self.num_actions = num_actions

        # PIE estimator
        self.estimator = PIEEstimator(
            input_dim_prop=num_proprio,
            input_dim_depth=num_scan,
            output_dim_velocity=velocity_dim,
            output_dim_foot_clearance=foot_clearance_dim,
            output_dim_height_map=height_map_dim,
            output_dim_latent=latent_dim,
            hidden_dim=512,
            hist_len=hist_len
        )
        
        # Actor
        self.actor = PIEActor(
            num_proprio=num_proprio,
            num_actions=num_actions,
            velocity_dim=velocity_dim,
            foot_clearance_dim=foot_clearance_dim,
            height_map_dim=height_map_dim,
            latent_dim=latent_dim,
            hidden_dims=actor_hidden_dims,
            activation=activation
        )
        
        # Critic - same as in ActorCriticRMA
        activation_fn = get_activation(activation)
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l+1]))
                critic_layers.append(activation_fn)
        self.critic = nn.Sequential(*critic_layers)
        
        # Action noise for exploration
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
    def act(self, proprio_obs, depth_images, prop_history, return_estimations=False):
        """Get actions from the actor critic based on proprio obs, depth images and history."""
        # Run the PIE estimator to get the state estimations
        base_velocity, foot_clearance, height_map_encoding, latent_vector, latent_mu, latent_logvar, next_state_pred = self.estimator(
            depth_images, prop_history
        )
        
        # Get actions using the actor with the estimated states
        actions_mean = self.actor(
            proprio_obs, 
            base_velocity, 
            foot_clearance, 
            height_map_encoding, 
            latent_vector
        )
        
        # Create distribution using the mean and std parameter
        self.distribution = Normal(actions_mean, self.std)
        
        # Sample actions from the distribution
        actions = self.distribution.sample()
        
        if return_estimations:
            return actions, base_velocity, foot_clearance, height_map_encoding, latent_vector
        else:
            return actions
            
    def act_inference(self, proprio_obs, depth_images, prop_history):
        """Inference mode (no sampling)"""
        with torch.no_grad():
            base_velocity, foot_clearance, height_map_encoding, latent_vector, _, _, _ = self.estimator(
                depth_images, prop_history
            )
            
            actions = self.actor(
                proprio_obs, 
                base_velocity, 
                foot_clearance, 
                height_map_encoding, 
                latent_vector
            )
            
        return actions
    
    def evaluate(self, critic_obs):
        """Get the value estimate from the critic network"""
        value = self.critic(critic_obs)
        return value
    
    def get_actions_log_prob(self, actions):
        """Get log probabilities of actions from the current distribution"""
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def reset(self, dones=None):
        """Method to reset internal states if needed between episodes"""
        # Reset the estimator GRU hidden states if they exist
        if hasattr(self.estimator, 'hidden_states') and self.estimator.hidden_states is not None:
            if dones is not None:
                # Reset only environments that are done
                mask = ~dones.bool().squeeze()
                batch_size = self.estimator.hidden_states.size(1)
                if mask.shape[0] < batch_size:  # Handle case with fewer dones than batch size
                    mask = torch.cat([mask, mask.new_ones(batch_size - mask.shape[0])])
                self.estimator.hidden_states[:, mask] = 0.0
            else:
                # Reset all environments
                self.estimator.hidden_states = None
    
    @property
    def action_mean(self):
        """Return the mean of the current action distribution"""
        return self.distribution.mean

    @property
    def action_std(self):
        """Return the std of the current action distribution"""
        return self.distribution.stddev
    
    @property
    def entropy(self):
        """Return the entropy of the current action distribution"""
        return self.distribution.entropy().sum(dim=-1)