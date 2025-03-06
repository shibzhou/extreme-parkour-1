import torch
import torch.nn as nn
import torch.nn.functional as F

class PIEEstimator(nn.Module):
    def __init__(self, 
                input_dim_prop, 
                input_dim_depth,  
                output_dim_velocity,
                output_dim_foot_clearance,
                output_dim_height_map,
                output_dim_latent,  
                hidden_dim=512,  
                hist_len=10):
        super(PIEEstimator, self).__init__()
        
        self.hist_len = hist_len
        self.input_dim_prop = input_dim_prop
        self.input_dim_depth = input_dim_depth
        
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 13 * 20, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim // 2)
        )
        
        self.prop_encoder = nn.Sequential(
            nn.Linear(input_dim_prop, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim // 2)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=1024),
            num_layers=2
        )

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            
        self.velocity_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim_velocity)
        )
        
        self.foot_clearance_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim_foot_clearance)
        )
        
        self.latent_mu = nn.Linear(hidden_dim, output_dim_latent)
        self.latent_logvar = nn.Linear(hidden_dim, output_dim_latent)

        self.height_map_decoder = nn.Sequential(
            nn.Linear(output_dim_latent, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim_latent)
        )
        
        self.successor_decoder = nn.Sequential(
            nn.Linear(output_dim_latent, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim_prop)
        )

        self.hidden_states = None
        
    def reset_hidden_states(self, batch_size, device):
        self.hidden_states = torch.zeros(1, batch_size, 512, device=device)
    
    def encode(self, depth_images, prop_history):
        batch_size = depth_images.shape[0]
        
        depth_features = self.depth_encoder(depth_images)
        
        prop_features = self.prop_encoder(prop_history.view(batch_size * self.hist_len, -1))
        prop_features = prop_features.view(batch_size, self.hist_len, -1)
        prop_features = prop_features[:, -1, :] 
        
        combined_features = torch.cat([depth_features, prop_features], dim=1)
        
        transformed_features = self.transformer_encoder(combined_features.unsqueeze(1))
        
        if self.hidden_states is None:
            self.reset_hidden_states(batch_size, depth_images.device)
        
        gru_out, self.hidden_states = self.gru(transformed_features, self.hidden_states)
        
        return gru_out.squeeze(1)
    
    def forward(self, depth_images, prop_history):
        encoded_features = self.encode(depth_images, prop_history)

        # explicit_estimates = self.explicit_estimator(encoded_features)
        
        base_velocity = self.velocity_estimator(encoded_features)
        foot_clearance = self.foot_clearance_estimator(encoded_features)

        latent_mu = self.latent_mu(encoded_features)
        latent_logvar = self.latent_logvar(encoded_features)

        if self.training:
            std = torch.exp(0.5 * latent_logvar)
            eps = torch.randn_like(std)
            latent_vector = latent_mu + eps * std
        else:
            latent_vector = latent_mu

        height_map_estimate = self.height_map_decoder(latent_vector)
        
        next_state_estimate = self.successor_decoder(latent_vector)
        
        return base_velocity, foot_clearance, height_map_estimate, latent_vector, latent_mu, latent_logvar, next_state_estimate    
    def compute_loss(self, explicit_estimates, height_map_estimate, next_state_estimate, 
                      latent_mu, latent_logvar, true_explicit, true_height_map, true_next_state):
        kl_loss = -0.5 * torch.sum(1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp())
        
        explicit_loss = F.mse_loss(explicit_estimates, true_explicit)
        height_map_loss = F.mse_loss(height_map_estimate, true_height_map)
        next_state_loss = F.mse_loss(next_state_estimate, true_next_state)
        
        total_loss = kl_loss + explicit_loss + height_map_loss + next_state_loss
        
        return total_loss, explicit_loss, height_map_loss, next_state_loss, kl_loss