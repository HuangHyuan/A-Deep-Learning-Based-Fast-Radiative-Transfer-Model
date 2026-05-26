import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class AdaptiveGradientConstraint(nn.Module):
    """
    Adaptive Gradient Constraint Layer: Dynamically controls gradients based on hydrometeor content.
    Ensures that zero-value hydrometeors (clear sky areas) do not contribute to gradient updates.
    """
    def __init__(self, threshold=0):
        super().__init__()
        self.threshold = threshold
        
    def forward(self, x_hydro):
        # x_hydro: [batch, num_layers, hydro_features]
        batch_size, num_layers, hydro_dim = x_hydro.shape
        
        # Create mask: 1 where hydrometeors exist (> threshold), 0 otherwise
        mask = (x_hydro > self.threshold).float() # [batch, num_layers, hydro_dim]
        
        # Apply mask: Output 0 for clear sky, keep original value for cloudy areas
        constrained_hydro = x_hydro * mask
        
        # Save mask for potential backward hook usage
        self.mask = mask
        
        return constrained_hydro, mask
    
    def backward_hook(self, grad):
        if hasattr(self, 'mask'):
            # Zero out gradients for positions without hydrometeors
            return grad * self.mask
        return grad

def get_sinusoidal_position_encoding(max_len, d_model):
    """
    Generate sinusoidal position encoding table
    :param max_len: Maximum sequence length (vertical layers)
    :param d_model: Embedding dimension (hidden_size)
    :return: Positional encoding of shape (1, max_len, d_model)
    """
    position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)
    
    pe = torch.zeros(1, max_len, d_model)  # (1, max_len, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    
    return pe  # No gradient needed when registered as buffer

class SceneAwareOutput(nn.Module):
    """
    Scene-Aware Output Layer: Adjusts output transformation based on cloud conditions.
    Uses a gating mechanism to balance clear-sky and cloudy-sky predictions.
    """
    def __init__(self, hidden_dim, output_dim, context_dim):
        super().__init__()
        self.fc_clear = nn.Linear(hidden_dim, 1)
        self.fc_cloudy = nn.Linear(hidden_dim, 1)
        # Final fully connected layer mapping to brightness temperatures (num_channels)
        self.fc_BT = nn.Sequential(
            nn.Linear(37, output_dim))
        self.context_dim = context_dim

        # Scene selection gating network
        self.scene_gate = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, cloud_context):
        batch_size = x.size(0)
        
        if cloud_context is not None:
            # Calculate gate weights based on cloud context
            gate_weights = self.scene_gate(cloud_context)
            
            # Mix clear and cloudy outputs based on gate weights
            output_clear = self.fc_clear(x)
            output_cloudy = self.fc_cloudy(x)
            
            # Weighted sum: [B, T, 1] * weight + [B, T, 1] * weight
            output_mixed = (gate_weights[:, 0].unsqueeze(-1) * output_clear.squeeze(-1) +
                          gate_weights[:, 1].unsqueeze(-1) * output_cloudy.squeeze(-1))
        else:
            # Fallback for clear sky mode (no cloud context)
            gate_weights = self.scene_gate(torch.zeros(batch_size, self.context_dim).to(x.device))
            output_clear = self.fc_clear(x)

            output_mixed = gate_weights[:, 0].unsqueeze(-1) * output_clear.squeeze(-1) 
            
        return self.fc_BT(output_mixed)
    
    
class AttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.0):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Input shape: (B, T, D)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, hidden_size)
        residual = x
        
        # Self-Attention + Residual Connection + LayerNorm
        x, _ = self.attention(x, x, x)  # Q=K=V=x
        x = self.norm1(residual + self.dropout(x))
        
        # Feed-Forward + Residual Connection + LayerNorm
        residual = x
        x = self.ffn(x)
        x = self.norm2(residual + self.dropout(x))
        
        return x
        
class UnifiedAllSkyRTM_Transformer(nn.Module):
    def __init__(
        self,
        num_layers=37,
        num_channels=22,
        atm_dim=3,
        surf_dim=5,
        hydro_dim=4,
        hidden_dim=256,
        nhead=4,
        num_transformer_layers=2,
        use_adaptive_constraint=True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_adaptive_constraint = use_adaptive_constraint
        
        if use_adaptive_constraint:
            self.adaptive_constraint = AdaptiveGradientConstraint()
        
        # --- Encoders ---
        # Atmosphere Encoder: atm_dim -> hidden_dim//4
        self.atm_encoder = nn.Sequential(
            nn.Linear(atm_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4)
        )
        # Surface Encoder: surf_dim -> hidden_dim//4
        self.surf_encoder = nn.Sequential(
            nn.Linear(surf_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4)
        )
        # Hydrometeor Encoder: hydro_dim -> hidden_dim//2
        self.hydro_encoder = nn.Sequential(
            nn.Linear(hydro_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2)
        )
        
        # Transformer Encoder Stack
        self.transformer = nn.ModuleList([
            AttentionBlock(hidden_dim, num_heads=nhead) 
            for _ in range(num_transformer_layers)
        ])
        
        # Positional encoding buffer
        self.register_buffer(
            'pos_encoding',
            get_sinusoidal_position_encoding(num_layers, hidden_dim)
        )
        
        # Output Layer
        self.scene_aware_output = SceneAwareOutput(
            hidden_dim=hidden_dim,
            output_dim=num_channels,
            context_dim=hydro_dim
        )

    def forward(self, x_atm, x_surf, x_hydro=None):
        batch_size = x_atm.size(0)
        device = x_atm.device

        # --- 1. Input Preprocessing & Detaching Constants ---
        # Separate variables (trainable) and constants (fixed physics)
        x_var1   = x_atm[:, :, 0].unsqueeze(-1)             # Water Vapor
        x_var2   = x_atm[:, :, 1].unsqueeze(-1)             # Temperature
        x_const3 = x_atm[:, :, 2].unsqueeze(-1).detach()    # Pressure (detached)

        # Recombine inputs
        x_atm = torch.cat([x_var1, x_var2, x_const3], dim=-1)  # (B, T, 3)
        atm_encoded = self.atm_encoder(x_atm)                  # (B, T, hidden_dim//4)

        # Surface processing: separate variables and constants (TCLW, TCRW)
        sfc_var   = x_surf[:, :3]
        sfc_const = x_surf[:, 3:].detach()    

        x_surf = torch.cat([sfc_var, sfc_const], dim=-1)  # (B, 5)
        # Expand surface features to match vertical layers (B, T, hidden_dim//4)
        surf_encoded = self.surf_encoder(x_surf).unsqueeze(1).repeat(1, self.num_layers, 1)
        
        # --- 2. Determine Mode: Clear vs. Cloudy ---
        is_clear_mode = False
        
        if x_hydro is None:
            is_clear_mode = True
        # Note: The explicit check for all-zeros is commented out in original logic, 
        # relying on None check. Uncomment below if zero-tensor check is needed.
        # else:
        #     if torch.sum(torch.abs(x_hydro)) == 0.0:
        #         is_clear_mode = True        
        
        if is_clear_mode:
            # === Clear Sky Mode ===
            # Create zero-filled hydrometeor encoding
            hydro_out_features = self.hydro_encoder[-1].out_features
            hydro_encoded = torch.zeros(batch_size, self.num_layers, hydro_out_features, device=device, dtype=x_atm.dtype)
            cloud_context = None
        else:
            # === Cloudy Mode ===
            if self.use_adaptive_constraint:
                constrained_hydro, _ = self.adaptive_constraint(x_hydro)
            else:
                constrained_hydro = x_hydro

            # Ensure vertical dimension matches num_layers (Dynamic padding/truncation)
            if constrained_hydro.shape[1] != self.num_layers:
                pad_len = self.num_layers - constrained_hydro.shape[1]
                if pad_len > 0:
                    constrained_hydro = F.pad(constrained_hydro, (0, 0, 0, pad_len), mode='constant', value=0.0)
                elif pad_len < 0:
                    constrained_hydro = constrained_hydro[:, :self.num_layers, :]
            
            # Encode hydrometeors
            hydro_encoded = self.hydro_encoder(constrained_hydro) # (B, T, hidden_dim//2)
            
            # Calculate Cloud Context vector
            # Average pooling over vertical dimension -> (B, hydro_dim)
            cloud_context = constrained_hydro.mean(dim=1)

        # --- 3. Feature Fusion ---
        # atm_encoded:   (B, T, H/4)
        # surf_encoded:  (B, T, H/4)
        # hydro_encoded: (B, T, H/2)
        # Total Dim: H/4 + H/4 + H/2 = H (hidden_dim)
        combined = torch.cat([atm_encoded, surf_encoded, hydro_encoded], dim=-1)

        # Add positional encoding
        src = combined + self.pos_encoding

        # --- 4. Transformer Processing ---
        for block in self.transformer:
            src = block(src)  # (B, T, hidden_dim)

        # --- 5. Output Prediction ---
        # Residual connection: add original combined features to transformer output
        tb_output = self.scene_aware_output(src + combined, cloud_context)
        return tb_output