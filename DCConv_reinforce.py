import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# FC Encoder
# ---------------------------------------------------------------------------
class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (batch, num_nodes, input_dim)
        batch, num_nodes, _ = x.size()
        x = x.view(-1, x.size(-1))
        out = self.network(x)
        out = F.relu(out)
        return out.view(batch, num_nodes, -1)

# ---------------------------------------------------------------------------
# Dilated Causal Convolution Block: A dilated convolution block with gated activation, residual and skip connections.
# ---------------------------------------------------------------------------
class DilatedCausalConvolutionBlock(nn.Module):
    def __init__(self, hidden_dim, dilation, kernel_size=2):
        super().__init__()
        # To keep the same output length, we use appropriate padding.
        self.padding = (kernel_size - 1) * dilation
        self.filter_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size,
                                     dilation=dilation, padding=self.padding)
        self.gate_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size,
                                   dilation=dilation, padding=self.padding)
        self.residual_conv = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.skip_conv = nn.Conv1d(hidden_dim, hidden_dim, 1)
    
    def forward(self, x):
        # x: (batch, hidden_dim, seq_len)
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        out = filter_out * gate_out  
        skip = self.skip_conv(out)
        residual = self.residual_conv(out)
        # Remove extra padding at the end so that residual matches x’s length.
        if self.padding != 0:
            residual = residual[:, :, :-self.padding]
            skip = skip[:, :, :-self.padding]
        # Add residual connection (assuming x and residual now share shape)
        x = x[:, :, :residual.size(2)] + residual
        return x, skip

# ---------------------------------------------------------------------------
# Hierarchical Decoder
# ---------------------------------------------------------------------------
class DilatedCausalConvolutionHierarchicalDecoder(nn.Module):
    def __init__(self, hidden_dim, num_vehicle_types, num_demand_types, num_help_hubs, 
                 travel_time_mean=1.0, travel_time_std=0.1, num_DCC_layers=4):
        super().__init__()
        self.num_vehicle_types = num_vehicle_types
        self.num_demand_types = num_demand_types
        self.num_help_hubs = num_help_hubs
        self.hidden_dim = hidden_dim
        
        # Dynamic state components
        self.location_embedding = nn.Embedding(1 + num_help_hubs, hidden_dim)
        self.capacity_projector = nn.Linear(num_vehicle_types * num_demand_types, hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # For main hub: compute pointer score using its fused feature.
        self.main_hub_pointer = nn.Linear(hidden_dim, 1)
        
        # For help hubs: process using a DilatedCausalConvolution stack.
        self.DilatedCausalConvolution_layers = nn.ModuleList()
        dilations = [2 ** i for i in range(num_DCC_layers)]
        for dilation in dilations:
            self.DilatedCausalConvolution_layers.append(DilatedCausalConvolutionBlock(hidden_dim, dilation=dilation))
        
        # After stacking, combine skip connections and project to pointer scores.
        self.skip_conv = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.skip_projection = nn.Linear(hidden_dim, 1)  # yields a score per help hub
        
        # For each help hub, decode vehicle–demand logits.
        self.vd_decoder = nn.Linear(hidden_dim, num_vehicle_types * num_demand_types)
        
        # Travel time projector (to incorporate travel time dynamics)
        self.travel_time_projector = nn.Linear(2, hidden_dim)
        self.travel_time_mean = travel_time_mean
        self.travel_time_std = travel_time_std
    
    def forward(self, node_embeddings, node_coords, current_location, vehicle_capacities):
        """
        node_embeddings: (batch, num_nodes, hidden_dim); node 0 is main hub, nodes 1: are help hubs.
        node_coords: (batch, num_nodes, 2)
        current_location: (batch,) long tensor (indices into nodes)
        vehicle_capacities: (batch, num_vehicle_types, num_demand_types)
        """
        batch, num_nodes, hidden_dim = node_embeddings.size()
        
        # Compute dynamic state.
        loc_emb = self.location_embedding(current_location)  # (batch, hidden_dim)
        cap_flat = vehicle_capacities.view(batch, -1)
        cap_emb = self.capacity_projector(cap_flat)           # (batch, hidden_dim)
        dynamic_state = self.fusion(torch.cat([loc_emb, cap_emb], dim=-1))  # (batch, hidden_dim)
        
        # Fuse dynamic state with node embeddings.
        fused = node_embeddings + dynamic_state.unsqueeze(1)  # (batch, num_nodes, hidden_dim)
        
        # Incorporate travel time features.
        batch_indices = torch.arange(batch, device=node_coords.device).unsqueeze(1)
        current_coords = node_coords[batch_indices, current_location.unsqueeze(1)].squeeze(1)  # (batch, 2)
        current_coords_exp = current_coords.unsqueeze(1).expand(-1, num_nodes, -1)
        distances = torch.norm(node_coords - current_coords_exp, dim=-1, p=2)  # (batch, num_nodes)
        travel_time_est = distances * self.travel_time_mean
        uncertainty = distances * self.travel_time_std
        tt_features = torch.stack([travel_time_est, uncertainty], dim=-1)  # (batch, num_nodes, 2)
        tt_embedding = F.relu(self.travel_time_projector(tt_features))      # (batch, num_nodes, hidden_dim)
        fused = fused + tt_embedding
        
        # ----- Main Hub Processing -----
        main_hub_feature = fused[:, 0, :]  # (batch, hidden_dim)
        main_hub_score = self.main_hub_pointer(main_hub_feature)  # (batch, 1)
        main_hub_prob = torch.sigmoid(main_hub_score)  # (batch, 1)
        
        # ----- Help Hubs Processing via DilatedCausalConvolution Stack -----
        help_features = fused[:, 1:, :]  # (batch, num_help_hubs, hidden_dim)
        # Transpose to (batch, hidden_dim, num_help_hubs) for 1D convolutions.
        x = help_features.transpose(1, 2)
        skip_connections = []
        for layer in self.DilatedCausalConvolution_layers:
            x, skip = layer(x)
            skip_connections.append(skip)
        # Sum skip connections.
        skip_sum = sum(skip_connections)
        # Process the skip-sum with a 1x1 convolution.
        refined = F.relu(self.skip_conv(skip_sum))  # (batch, hidden_dim, num_help_hubs)
        # Transpose back: (batch, num_help_hubs, hidden_dim)
        refined = refined.transpose(1, 2)
        
        # Compute pointer scores for help hubs.
        help_hub_pointer_scores = self.skip_projection(refined).squeeze(-1)  # (batch, num_help_hubs)
        help_hub_probs = F.softmax(help_hub_pointer_scores, dim=-1)  # (batch, num_help_hubs)
        
        # Decode vehicle–demand logits for each help hub.
        vd_logits = self.vd_decoder(refined)  # (batch, num_help_hubs, num_vehicle_types*num_demand_types)
        vd_probs = F.softmax(vd_logits, dim=-1)
        
        # Combine the help-hub pointer with vehicle–demand probabilities.
        combined_help = help_hub_probs.unsqueeze(-1) * vd_probs  # (batch, num_help_hubs, num_vehicle_types*num_demand_types)
        combined_help = combined_help.view(batch, -1)  # flatten to (batch, num_help_hubs * num_vehicle_types*num_demand_types)
        
        # ----- Final Action Distribution -----
        # Action 0 corresponds to the main hub; actions 1: correspond to help hubs' vehicle–demand pairs.
        final_probs = torch.cat([main_hub_prob, combined_help], dim=-1)  # (batch, 1 + num_help_hubs * num_vehicle_types*num_demand_types)
        EPS = 1e-6
        final_logits = torch.log(final_probs + EPS)
        return final_logits

# ---------------------------------------------------------------------------
# Critic Network
# ---------------------------------------------------------------------------
class CriticNetwork(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, pooled_static, dynamic_state):
        x = torch.cat([pooled_static, dynamic_state], dim=-1)
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value.squeeze(-1)

# ---------------------------------------------------------------------------
# Overall Model with the DilatedCausalConvolution Decoder
# ---------------------------------------------------------------------------
class HierarchicalReinforce(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=256, encoder_layers=2, n_heads=2,
                 num_help_hubs=5, num_vehicle_types=3, num_demand_types=3,
                 travel_time_mean=1.0, travel_time_std=0.1, num_DCC_layers=4):
        super().__init__()
        self.num_help_hubs = num_help_hubs
        self.num_vehicle_types = num_vehicle_types
        self.num_demand_types = num_demand_types
        self.total_actions = 1 + num_help_hubs * num_vehicle_types * num_demand_types
        
        self.encoder = SimpleEncoder(input_dim, hidden_dim, encoder_layers)
        # Global mixing with self-attention.
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        self.decoder = DilatedCausalConvolutionHierarchicalDecoder(hidden_dim, num_vehicle_types, num_demand_types, num_help_hubs,
                                                  travel_time_mean=travel_time_mean, travel_time_std=travel_time_std,
                                                  num_DCC_layers=num_DCC_layers)
        self.critic = CriticNetwork(hidden_dim)
    
    def forward(self, main_hub_xy, help_hub_xy, demands, current_location, vehicle_capacities):
        """
        main_hub_xy: (batch, 1, 2)
        help_hub_xy: (batch, num_help_hubs, 2)
        demands: (batch, num_help_hubs, num_demand_types)
        current_location: (batch,) indices into nodes (0 for main hub)
        vehicle_capacities: (batch, num_vehicle_types, num_demand_types)
        """
        batch_size = main_hub_xy.size(0)
        device = main_hub_xy.device
        
        # Build node features: for the main hub we append zeros for demands.
        zeros = torch.zeros(batch_size, 1, self.num_demand_types, device=device)
        main_features = torch.cat([main_hub_xy, zeros], dim=-1)
        help_features = torch.cat([help_hub_xy, demands], dim=-1)
        node_features = torch.cat([main_features, help_features], dim=1)
        node_coords = node_features[..., :2]
        
        # Encode static node features.
        node_embeddings = self.encoder(node_features)
        # Global mixing via self-attention.
        node_embeddings, _ = self.attention(node_embeddings, node_embeddings, node_embeddings)
        
        logits = self.decoder(node_embeddings, node_coords, current_location, vehicle_capacities)
        
        # Critic branch: reuse dynamic state computation.
        loc_emb = self.decoder.location_embedding(current_location)
        cap_flat = vehicle_capacities.view(batch_size, -1)
        cap_emb = self.decoder.capacity_projector(cap_flat)
        dynamic_state = self.decoder.fusion(torch.cat([loc_emb, cap_emb], dim=-1))
        pooled_static = node_embeddings.mean(dim=1)
        value = self.critic(pooled_static, dynamic_state)
        
        return logits, value

    def act(self, main_hub_xy, help_hub_xy, demands, current_location, vehicle_capacities):
        logits, value = self(main_hub_xy, help_hub_xy, demands, current_location, vehicle_capacities)
        probs = F.softmax(logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        # Return action on CPU to align with the environment.
        return action.cpu(), log_prob, value
