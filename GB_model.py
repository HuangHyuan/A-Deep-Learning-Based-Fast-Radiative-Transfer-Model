"""Radiative-transfer emulators used in the manuscript.

Implemented architectures:
- Transformer (``UnifiedAllSkyRTM_Transformer``)
- bidirectional LSTM (``UnifiedAllSkyRTM_BiLSTM``)
- MLP (``UnifiedAllSkyRTM_MLP``)
- simple vertical LSTM (``SimpleVerticalLSTM``)
"""

import torch.nn as nn
import torch
import math
import torch.nn.functional as F
class AdaptiveGradientConstraint(nn.Module):
    """Mask hydrometeor features so levels without hydrometeors receive no gradient."""
    def __init__(self, threshold=0):
        super().__init__()
        self.threshold = threshold
    def forward(self, x_hydro):
        # x_hydro: [batch, num_layers, hydro_features]
        batch_size, num_layers, hydro_dim = x_hydro.shape
        mask = (x_hydro > self.threshold).float() # [batch, num_layers, hydro_dim]
        constrained_hydro = x_hydro * mask
        self.mask = mask
        return constrained_hydro, mask
    def backward_hook(self, grad):
        if hasattr(self, 'mask'):
            return grad * self.mask
        return grad
def get_sinusoidal_position_encoding(max_len, d_model):
    """Return sinusoidal positional encodings for the vertical sequence."""
    position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)
    pe = torch.zeros(1, max_len, d_model)  # (1, max_len, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe
class SceneAwareOutput_Sequence(nn.Module):
    """Gate clear-sky and cloudy-sky tokens before mapping to brightness temperatures."""
    def __init__(self, hidden_dim, output_dim, context_dim, n_layers=37):
        super().__init__()
        self.fc_clear = nn.Linear(hidden_dim, 1)
        self.fc_cloudy = nn.Linear(hidden_dim, 1)
        self.fc_BT = nn.Sequential(
            nn.Linear(n_layers, output_dim))
        self.context_dim = context_dim
        self.scene_gate = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )
    def forward(self, x, cloud_context):
        batch_size = x.size(0)
        if cloud_context is not None:
            gate_weights = self.scene_gate(cloud_context)
            output_clear = self.fc_clear(x)
            output_cloudy = self.fc_cloudy(x)
            output_mixed = (gate_weights[:, 0].unsqueeze(-1) * output_clear.squeeze(-1) +
                          gate_weights[:, 1].unsqueeze(-1) * output_cloudy.squeeze(-1))
        else:
            gate_weights = self.scene_gate(torch.zeros(batch_size, self.context_dim).to(x.device))
            output_clear = self.fc_clear(x)
            output_mixed = gate_weights[:, 0].unsqueeze(-1) * output_clear.squeeze(-1)
        return self.fc_BT(output_mixed)
class SequenceMLPBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(self, x):
        # x: [Batch, num_layers, hidden_dim]
        return self.net(x)
class UnifiedAllSkyRTM_MLP(nn.Module):
    """All-sky MLP emulator with scene-aware output gating."""
    def __init__(
        self,
        num_layers=37,
        num_channels=22,
        atm_dim=3,
        surf_dim=5,
        hydro_dim=4,
        hidden_dim=12,
        num_mlp_blocks=3,
        use_adaptive_constraint=True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hydro_dim = hydro_dim
        self.hidden_dim = hidden_dim
        self.use_adaptive_constraint = use_adaptive_constraint
        if use_adaptive_constraint:
            self.adaptive_constraint = AdaptiveGradientConstraint()
        self.atm_encoder = nn.Sequential(
            nn.Linear(atm_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4)
        )
        self.surf_encoder = nn.Sequential(
            nn.Linear(surf_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4)
        )
        self.hydro_encoder = nn.Sequential(
            nn.Linear(hydro_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2)
        )
        self.mlp_blocks = nn.ModuleList([
            SequenceMLPBlock(hidden_dim * num_layers) for _ in range(num_mlp_blocks)
        ])
        self.scene_aware_output = SceneAwareOutput_Sequence(
            hidden_dim=hidden_dim,
            output_dim=num_channels,
            context_dim=hydro_dim,
            n_layers=num_layers,
        )
    def forward(self, x_atm, x_surf, x_hydro=None):
        batch_size = x_atm.size(0)
        device = x_atm.device
        x_var1   = x_atm[:, :, 0].unsqueeze(-1)
        x_var2   = x_atm[:, :, 1].unsqueeze(-1)
        x_const3 = x_atm[:, :, 2].unsqueeze(-1).detach()
        x_atm = torch.cat([x_var1, x_var2, x_const3], dim=-1)  # (B, T, 4)
        atm_encoded = self.atm_encoder(x_atm)
        sfc_var   = x_surf[:, :3]
        sfc_const = x_surf[:, 3:].detach()    # TCLW
        x_surf = torch.cat([sfc_var, sfc_const], dim=-1)  # (B, T, 4)
        surf_encoded = self.surf_encoder(x_surf).unsqueeze(1).repeat(1, self.num_layers, 1)
        is_clear_mode = False
        if x_hydro is None:
            is_clear_mode = True
        # else:
        #     if torch.sum(torch.abs(x_hydro)) == 0.0:
        #         is_clear_mode = True
        if is_clear_mode:
            hydro_out_features = self.hydro_encoder[-1].out_features
            hydro_encoded = torch.zeros(batch_size, self.num_layers, hydro_out_features, device=device, dtype=x_atm.dtype)
            cloud_context = None
        else:
            if self.use_adaptive_constraint:
                constrained_hydro, _ = self.adaptive_constraint(x_hydro)
            else:
                constrained_hydro = x_hydro
            if constrained_hydro.shape[1] != self.num_layers:
                pad_len = self.num_layers - constrained_hydro.shape[1]
                if pad_len > 0:
                    constrained_hydro = F.pad(constrained_hydro, (0, 0, 0, pad_len), mode='constant', value=0.0)
                elif pad_len < 0:
                    constrained_hydro = constrained_hydro[:, :self.num_layers, :]
            hydro_encoded = self.hydro_encoder(constrained_hydro)
            cloud_context = constrained_hydro.mean(dim=1)
        combined = torch.cat([atm_encoded, surf_encoded, hydro_encoded], dim=-1)  # [B, L, D]
        features = combined.reshape(-1,self.hidden_dim*self.num_layers)
        for mlp_block in self.mlp_blocks:
            features = mlp_block(features)
        tb_output = self.scene_aware_output(features.reshape(-1,self.num_layers,self.hidden_dim)+combined, cloud_context)
        return tb_output
class BiLSTMBlock(nn.Module):
    """Residual bidirectional-LSTM block used by the BiLSTM emulator."""
    def __init__(self, hidden_size, dropout=0.0):
        super(BiLSTMBlock, self).__init__()
        self.LSTMn= nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size // 2,
                batch_first=True,
                bidirectional=True
            )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
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
        # Self-Attention + Residual + Norm
        x, _ = self.LSTMn(x)  # Q=K=V=x
        x = self.norm1(residual + self.dropout(x))
        # Feed-Forward + Residual + Norm
        residual = x
        x = self.ffn(x)
        x = self.norm2(residual + self.dropout(x))
        return x
class UnifiedAllSkyRTM_BiLSTM(nn.Module):
    """All-sky bidirectional-LSTM emulator for vertical atmospheric profiles."""
    def __init__(
        self,
        num_layers=37,
        num_channels=22,
        atm_dim=3,
        surf_dim=5,
        hydro_dim=4,
        hidden_dim=180,
        num_bi_lstm_blocks=2,
        use_adaptive_constraint=True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hydro_dim = hydro_dim
        self.use_adaptive_constraint = use_adaptive_constraint
        if use_adaptive_constraint:
            self.adaptive_constraint = AdaptiveGradientConstraint()
        self.atm_encoder = nn.Sequential(
            nn.Linear(atm_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4)
        )
        self.surf_encoder = nn.Sequential(
            nn.Linear(surf_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4)
        )
        self.hydro_encoder = nn.Sequential(
            nn.Linear(hydro_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2)
        )
        self.bi_lstm_blocks = nn.ModuleList([
            BiLSTMBlock(hidden_dim)
            for _ in range(num_bi_lstm_blocks)
        ])
        self.scene_aware_output = SceneAwareOutput_Sequence(
            hidden_dim=hidden_dim,
            output_dim=num_channels,
            context_dim=hydro_dim,
            n_layers=num_layers,
        )
    def forward(self, x_atm, x_surf, x_hydro=None):
        batch_size = x_atm.size(0)
        device = x_atm.device
        x_var1   = x_atm[:, :, 0].unsqueeze(-1)
        x_var2   = x_atm[:, :, 1].unsqueeze(-1)
        x_const3 = x_atm[:, :, 2].unsqueeze(-1).detach()
        x_atm = torch.cat([x_var1, x_var2, x_const3], dim=-1)  # (B, T, 4)
        atm_encoded = self.atm_encoder(x_atm)
        sfc_var   = x_surf[:, :3]
        sfc_const = x_surf[:, 3:].detach()    # TCLW
        x_surf = torch.cat([sfc_var, sfc_const], dim=-1)  # (B, T, 4)
        surf_encoded = self.surf_encoder(x_surf).unsqueeze(1).repeat(1, self.num_layers, 1)
        is_clear_mode = False
        if x_hydro is None:
            is_clear_mode = True
        # else:
        #     if torch.sum(torch.abs(x_hydro)) == 0.0:
        #         is_clear_mode = True
        if is_clear_mode:
            hydro_out_features = self.hydro_encoder[-1].out_features
            hydro_encoded = torch.zeros(batch_size, self.num_layers, hydro_out_features, device=device, dtype=x_atm.dtype)
            cloud_context = None
        else:
            if self.use_adaptive_constraint:
                constrained_hydro, _ = self.adaptive_constraint(x_hydro)
            else:
                constrained_hydro = x_hydro
            if constrained_hydro.shape[1] != self.num_layers:
                pad_len = self.num_layers - constrained_hydro.shape[1]
                if pad_len > 0:
                    constrained_hydro = F.pad(constrained_hydro, (0, 0, 0, pad_len), mode='constant', value=0.0)
                elif pad_len < 0:
                    constrained_hydro = constrained_hydro[:, :self.num_layers, :]
            hydro_encoded = self.hydro_encoder(constrained_hydro)
            cloud_context = constrained_hydro.mean(dim=1)
        combined = torch.cat([atm_encoded, surf_encoded, hydro_encoded], dim=-1)
        # BiLSTM processing
        lstm_out = combined
        for lstm in self.bi_lstm_blocks:
            lstm_out = lstm(lstm_out)
        tb_output = self.scene_aware_output(lstm_out+combined, cloud_context)
        return tb_output
class AttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.0):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # (B, T, D)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
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
        # Self-Attention + Residual + Norm
        x, _ = self.attention(x, x, x)  # Q=K=V=x
        x = self.norm1(residual + self.dropout(x))
        # Feed-Forward + Residual + Norm
        residual = x
        x = self.ffn(x)
        x = self.norm2(residual + self.dropout(x))
        return x
class UnifiedAllSkyRTM_Transformer(nn.Module):
    """All-sky Transformer emulator using the vertical profile as a sequence."""
    def __init__(
        self,
        num_layers=37,
        num_channels=22,
        atm_dim=3,
        surf_dim=5,
        hydro_dim=4,
        hidden_dim=160,
        nhead=4,
        num_transformer_layers=3,
        use_adaptive_constraint=True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_adaptive_constraint = use_adaptive_constraint
        if use_adaptive_constraint:
            self.adaptive_constraint = AdaptiveGradientConstraint()
        self.atm_encoder = nn.Sequential(
            nn.Linear(atm_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4)
        )
        self.surf_encoder = nn.Sequential(
            nn.Linear(surf_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4)
        )
        self.hydro_encoder = nn.Sequential(
            nn.Linear(hydro_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2)
        )
        # Transformer encoder
        self.transformer = nn.ModuleList([
            AttentionBlock(hidden_dim, num_heads=nhead)
            for _ in range(num_transformer_layers)
        ])
        # Positional encoding
        self.register_buffer(
            'pos_encoding',
            get_sinusoidal_position_encoding(num_layers, hidden_dim)
        )
        self.scene_aware_output = SceneAwareOutput_Sequence(
            hidden_dim=hidden_dim,
            output_dim=num_channels,
            context_dim=hydro_dim,
            n_layers=num_layers,
        )
    def forward(self, x_atm, x_surf, x_hydro=None):
        batch_size = x_atm.size(0)
        device = x_atm.device
        x_var1   = x_atm[:, :, 0].unsqueeze(-1)
        x_var2   = x_atm[:, :, 1].unsqueeze(-1)
        x_const3 = x_atm[:, :, 2].unsqueeze(-1).detach()
        x_atm = torch.cat([x_var1, x_var2, x_const3], dim=-1)  # (B, T, 4)
        atm_encoded = self.atm_encoder(x_atm)
        sfc_var   = x_surf[:, :3]
        sfc_const = x_surf[:, 3:].detach()    # TCLW,TCRW
        x_surf = torch.cat([sfc_var, sfc_const], dim=-1)  # (B, T, 4)
        surf_encoded = self.surf_encoder(x_surf).unsqueeze(1).repeat(1, self.num_layers, 1)
        is_clear_mode = False
        if x_hydro is None:
            is_clear_mode = True
        # else:
        #     if torch.sum(torch.abs(x_hydro)) == 0.0:
        #         is_clear_mode = True
        if is_clear_mode:
            hydro_out_features = self.hydro_encoder[-1].out_features
            hydro_encoded = torch.zeros(batch_size, self.num_layers, hydro_out_features, device=device, dtype=x_atm.dtype)
            cloud_context = None
        else:
            if self.use_adaptive_constraint:
                constrained_hydro, _ = self.adaptive_constraint(x_hydro)
            else:
                constrained_hydro = x_hydro
            if constrained_hydro.shape[1] != self.num_layers:
                pad_len = self.num_layers - constrained_hydro.shape[1]
                if pad_len > 0:
                    constrained_hydro = F.pad(constrained_hydro, (0, 0, 0, pad_len), mode='constant', value=0.0)
                elif pad_len < 0:
                    constrained_hydro = constrained_hydro[:, :self.num_layers, :]
            hydro_encoded = self.hydro_encoder(constrained_hydro)
            cloud_context = constrained_hydro.mean(dim=1)
        # atm_encoded: (B, T, H/4)
        # surf_encoded: (B, T, H/4)
        # hydro_encoded: (B, T, H/2)
        # Total: H/4 + H/4 + H/2 = H (hidden_dim)
        combined = torch.cat([atm_encoded, surf_encoded, hydro_encoded], dim=-1)
        # Add positional encoding
        src = combined + self.pos_encoding
        # Transformer Encoder
        for block in self.transformer:
            src = block(src)  # (B, T, hidden_dim)
        tb_output = self.scene_aware_output(src+combined, cloud_context)
        return tb_output

class SimpleVerticalLSTM(nn.Module):
    """Two-layer vertical LSTM used as the compact revision baseline."""

    def __init__(
        self,
        num_layers=37,
        num_channels=22,
        atm_dim=3,
        surf_dim=5,
        hydro_dim=4,
        hidden_dim=128,
        num_lstm_layers=2,
        n_layers=None,
    ):
        super().__init__()
        if n_layers is not None:
            num_layers = n_layers
        self.num_layers = num_layers
        self.n_layers = num_layers
        self.num_channels = num_channels
        self.n_channels = num_channels
        self.input_proj = nn.Linear(atm_dim + surf_dim + hydro_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.head = nn.Linear(hidden_dim * num_layers, num_channels)

    def forward(self, x_atm, x_surf, x_hydro=None):
        batch_size, levels, _ = x_atm.shape
        if x_hydro is None:
            x_hydro = torch.zeros(
                batch_size, levels, 4, device=x_atm.device, dtype=x_atm.dtype
            )
        elif x_hydro.shape[1] < levels:
            # The original hydrometeor product has one fewer level.
            x_hydro = F.pad(x_hydro, (0, 0, 0, levels - x_hydro.shape[1]))
        elif x_hydro.shape[1] > levels:
            x_hydro = x_hydro[:, :levels]
        surface = x_surf.unsqueeze(1).expand(-1, levels, -1)
        features = torch.cat([x_atm, surface, x_hydro], dim=-1)
        # Iterate from the surface upward, then restore the original order.
        encoded, _ = self.lstm(torch.flip(self.input_proj(features), dims=[1]))
        encoded = torch.flip(encoded, dims=[1])
        return self.head(encoded.flatten(1))


# Backward- and reviewer-facing aliases.
SceneAwareOutput = SceneAwareOutput_Sequence
UnifiedAllSkyRTM = UnifiedAllSkyRTM_Transformer


def build_model(name, **kwargs):
    """Instantiate a supported architecture by its manuscript name."""
    key = name.lower().replace('-', '_')
    registry = {
        'transformer': UnifiedAllSkyRTM_Transformer,
        'bilstm': UnifiedAllSkyRTM_BiLSTM,
        'mlp': UnifiedAllSkyRTM_MLP,
        'simple_lstm': SimpleVerticalLSTM,
    }
    if key not in registry:
        raise ValueError(f'Unknown model: {name}')
    model = registry[key](**kwargs)
    # Shared experiment helpers read these attributes from every architecture.
    if not hasattr(model, 'n_layers'):
        model.n_layers = getattr(model, 'num_layers', kwargs.get('num_layers', kwargs.get('n_layers')))
    if not hasattr(model, 'n_channels'):
        model.n_channels = getattr(model, 'num_channels', kwargs.get('num_channels', kwargs.get('n_channels')))
    return model


__all__ = [
    'AdaptiveGradientConstraint',
    'SceneAwareOutput_Sequence',
    'SceneAwareOutput',
    'UnifiedAllSkyRTM',
    'UnifiedAllSkyRTM_Transformer',
    'UnifiedAllSkyRTM_BiLSTM',
    'UnifiedAllSkyRTM_MLP',
    'SimpleVerticalLSTM',
    'build_model',
]
