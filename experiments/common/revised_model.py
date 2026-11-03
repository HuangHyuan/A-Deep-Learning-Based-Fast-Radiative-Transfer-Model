"""Configurable radiative-transfer emulators for the reviewer-revision experiments.

The implementation keeps the vertical dimension as the sequence dimension and
supports both the original 37-level grid and a 137-level matched experiment.
"""

from __future__ import print_function

import math
import importlib
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from GB_model import SimpleVerticalLSTM


def get_sinusoidal_position_encoding(max_len: int, d_model: int) -> torch.Tensor:
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(1, max_len, d_model, dtype=torch.float32)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe


class AdaptiveGradientConstraint(nn.Module):
    """Apply a hard mask so zero hydrometeors cannot update the encoder."""

    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, x_hydro: torch.Tensor):
        mask = (x_hydro > self.threshold).to(dtype=x_hydro.dtype)
        return x_hydro * mask, mask


class AttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        batch_size, sequence_length, _ = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, sequence_length, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attention, value)
        attended = attended.transpose(1, 2).reshape(
            batch_size, sequence_length, self.hidden_size
        )
        x = self.norm1(residual + self.dropout(self.out_proj(attended)))
        residual = x
        x = self.ffn(x)
        return self.norm2(residual + self.dropout(x))


class SceneAwareOutput(nn.Module):
    """Mix clear/cloudy token gates and predict channels from the vertical axis."""

    def __init__(self, hidden_dim: int, n_layers: int, output_dim: int, context_dim: int = 4):
        super().__init__()
        self.n_layers = n_layers
        self.fc_clear = nn.Linear(hidden_dim, 1)
        self.fc_cloudy = nn.Linear(hidden_dim, 1)
        self.register_buffer(
            "clear_bias", torch.zeros(1, n_layers), persistent=True
        )
        self.scene_gate = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1),
        )
        self.fc_bt = nn.Linear(n_layers, output_dim)

    def forward(self, x: torch.Tensor, cloud_context: Optional[torch.Tensor]):
        clear_token = self.fc_clear(x).squeeze(-1) + self.clear_bias
        cloudy_token = self.fc_cloudy(x).squeeze(-1)
        if cloud_context is None:
            mixed = clear_token
        else:
            gate = self.scene_gate(cloud_context)
            mixed = gate[:, 0:1] * clear_token + gate[:, 1:2] * cloudy_token
        return self.fc_bt(mixed)


class TransformerRTM(nn.Module):
    """Vertical-sequence Transformer with a learned clear-sky branch."""

    def __init__(
        self,
        n_layers: int = 37,
        n_channels: int = 22,
        atm_dim: int = 3,
        surf_dim: int = 5,
        hydro_dim: int = 4,
        hidden_dim: int = 256,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        dropout: float = 0.0,
        use_adaptive_constraint: bool = True,
        clear_branch: str = "learned_token",
    ):
        super().__init__()
        if hidden_dim % 4 != 0:
            raise ValueError("hidden_dim must be divisible by 4")
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.hydro_dim = hydro_dim
        self.use_adaptive_constraint = use_adaptive_constraint
        self.clear_branch = clear_branch

        if use_adaptive_constraint:
            self.adaptive_constraint = AdaptiveGradientConstraint()

        half_hidden = hidden_dim // 2
        quarter_hidden = hidden_dim // 4
        self.atm_encoder = nn.Sequential(
            nn.Linear(atm_dim, half_hidden),
            nn.GELU(),
            nn.Linear(half_hidden, quarter_hidden),
        )
        self.surf_encoder = nn.Sequential(
            nn.Linear(surf_dim, half_hidden),
            nn.GELU(),
            nn.Linear(half_hidden, quarter_hidden),
        )
        self.hydro_encoder = nn.Sequential(
            nn.Linear(hydro_dim, half_hidden),
            nn.GELU(),
            nn.Linear(half_hidden, half_hidden),
        )
        if clear_branch == "learned_token":
            self.clear_hydro_token = nn.Parameter(torch.zeros(half_hidden))
        elif clear_branch == "zero":
            self.register_parameter("clear_hydro_token", None)
        else:
            raise ValueError("clear_branch must be 'learned_token' or 'zero'")

        self.transformer = nn.ModuleList(
            [
                AttentionBlock(hidden_dim, num_heads=nhead, dropout=dropout)
                for _ in range(num_transformer_layers)
            ]
        )
        self.register_buffer(
            "pos_encoding",
            get_sinusoidal_position_encoding(n_layers, hidden_dim),
        )
        self.scene_aware_output = SceneAwareOutput(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            output_dim=n_channels,
            context_dim=hydro_dim,
        )

    def forward(
        self,
        x_atm: torch.Tensor,
        x_surf: torch.Tensor,
        x_hydro: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = x_atm.shape[0]
        if x_atm.shape[1] != self.n_layers:
            raise ValueError(
                "Expected x_atm with {} levels, got {}".format(
                    self.n_layers, x_atm.shape[1]
                )
            )

        # These two channels are treated as fixed physical constraints in the
        # original implementation. The model therefore does not fit them.
        atm_input = torch.cat([x_atm[:, :, :2], x_atm[:, :, 2:3].detach()], dim=-1)
        atm_encoded = self.atm_encoder(atm_input)

        surf_input = torch.cat([x_surf[:, :3], x_surf[:, 3:].detach()], dim=-1)
        surf_encoded = self.surf_encoder(surf_input).unsqueeze(1).repeat(
            1, self.n_layers, 1
        )

        if x_hydro is None:
            if self.clear_branch == "learned_token":
                token = self.clear_hydro_token.view(1, 1, -1).expand(
                    batch_size, self.n_layers, -1
                )
                hydro_encoded = token
            else:
                hydro_encoded = torch.zeros(
                    batch_size,
                    self.n_layers,
                    self.hydro_encoder[-1].out_features,
                    device=x_atm.device,
                    dtype=x_atm.dtype,
                )
            cloud_context = None
        else:
            if x_hydro.shape[1] != self.n_layers:
                raise ValueError(
                    "x_hydro must use the same vertical length as x_atm in the "
                    "revised matched-grid experiments; got {} and {}".format(
                        x_hydro.shape[1], self.n_layers
                    )
                )
            constrained = (
                self.adaptive_constraint(x_hydro)[0]
                if self.use_adaptive_constraint
                else x_hydro
            )
            hydro_encoded = self.hydro_encoder(constrained)
            cloud_context = constrained.mean(dim=1)

        combined = torch.cat([atm_encoded, surf_encoded, hydro_encoded], dim=-1)
        src = combined + self.pos_encoding
        for block in self.transformer:
            src = block(src)
        return self.scene_aware_output(src + combined, cloud_context)


class MLPBaseline(nn.Module):
    """MLP baseline with the vertical dimension flattened."""

    def __init__(
        self,
        n_layers: int = 37,
        n_channels: int = 22,
        atm_dim: int = 3,
        surf_dim: int = 5,
        hydro_dim: int = 4,
        hidden_dim: int = 512,
    ):
        super().__init__()
        input_dim = n_layers * atm_dim + surf_dim + n_layers * hydro_dim
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_channels),
        )

    def forward(self, x_atm, x_surf, x_hydro=None):
        if x_hydro is None:
            x_hydro = torch.zeros(
                x_atm.shape[0],
                self.n_layers,
                4,
                device=x_atm.device,
                dtype=x_atm.dtype,
            )
        return self.net(torch.cat([x_atm.flatten(1), x_surf, x_hydro.flatten(1)], dim=1))


class BiLSTMBaseline(nn.Module):
    """Parameter-efficient bidirectional LSTM reference model."""

    def __init__(
        self,
        n_layers: int = 37,
        n_channels: int = 22,
        atm_dim: int = 3,
        surf_dim: int = 5,
        hydro_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.input_proj = nn.Linear(atm_dim + surf_dim + hydro_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if num_layers == 1 else 0.1,
        )
        self.head = nn.Linear(2 * hidden_dim * n_layers, n_channels)

    def forward(self, x_atm, x_surf, x_hydro=None):
        bsz, levels, _ = x_atm.shape
        if x_hydro is None:
            x_hydro = torch.zeros(bsz, levels, 4, device=x_atm.device, dtype=x_atm.dtype)
        surface = x_surf.unsqueeze(1).expand(-1, levels, -1)
        features = torch.cat([x_atm, surface, x_hydro], dim=-1)
        encoded, _ = self.lstm(self.input_proj(features))
        return self.head(encoded.flatten(1))





def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name: str, **kwargs) -> nn.Module:
    key = name.lower().rstrip("-")
    if key in {
        "transformer",
        "original_transformer",
        "bilstm",
        "bi_lstm",
        "original_bilstm",
        "mlp",
        "original_mlp",
    }:
        project_root = str(Path(__file__).resolve().parents[2])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        gb_model = importlib.import_module("GB_model")
        common = {
            "num_layers": kwargs["n_layers"],
            "num_channels": kwargs["n_channels"],
            "atm_dim": kwargs.get("atm_dim", 3),
            "surf_dim": kwargs.get("surf_dim", 5),
            "hydro_dim": kwargs.get("hydro_dim", 4),
        }
        if key in {"transformer", "original_transformer"}:
            model = gb_model.UnifiedAllSkyRTM_Transformer(
                **common,
                hidden_dim=kwargs.get("hidden_dim", 160),
                nhead=kwargs.get("nhead", 4),
                num_transformer_layers=kwargs.get("num_transformer_layers", 3),
            )
        elif key in {"bilstm", "bi_lstm", "original_bilstm"}:
            model = gb_model.UnifiedAllSkyRTM_BiLSTM(
                **common,
                hidden_dim=kwargs.get("hidden_dim", 180),
                num_bi_lstm_blocks=kwargs.get("num_layers", 2),
            )
        else:
            model = gb_model.UnifiedAllSkyRTM_MLP(
                **common,
                hidden_dim=kwargs.get("hidden_dim", 12),
                num_mlp_blocks=kwargs.get("num_layers", 3),
            )
        model.n_layers = kwargs["n_layers"]
        model.n_channels = kwargs["n_channels"]
        return model
    if key in {"configurable_transformer"}:
        return TransformerRTM(**kwargs)
    if key in {"configurable_mlp"}:
        mlp_keys = {
            k: v
            for k, v in kwargs.items()
            if k in {"n_layers", "n_channels", "atm_dim", "surf_dim", "hydro_dim", "hidden_dim"}
        }
        return MLPBaseline(**mlp_keys)
    if key in {"configurable_bilstm"}:
        lstm_keys = {
            k: v
            for k, v in kwargs.items()
            if k
            in {
                "n_layers",
                "n_channels",
                "atm_dim",
                "surf_dim",
                "hydro_dim",
                "hidden_dim",
                "num_layers",
            }
        }
        return BiLSTMBaseline(**lstm_keys)
    if key == "simple_lstm":
        return SimpleVerticalLSTM(
            n_layers=kwargs["n_layers"],
            num_channels=kwargs["n_channels"],
            atm_dim=kwargs.get("atm_dim", 3),
            surf_dim=kwargs.get("surf_dim", 5),
            hydro_dim=kwargs.get("hydro_dim", 4),
            hidden_dim=kwargs.get("hidden_dim", 128),
            num_lstm_layers=kwargs.get("num_layers", 2),
        )
    raise ValueError("Unknown model: {}".format(name))

__all__ = [
    "AdaptiveGradientConstraint",
    "AttentionBlock",
    "TransformerRTM",
    "MLPBaseline",
    "BiLSTMBaseline",
    "SimpleVerticalLSTM",
    "build_model",
    "count_parameters",
]
