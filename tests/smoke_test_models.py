"""Data-free checks of all architectures reported in the manuscript.

Run from the repository root:
    python tests/smoke_test_models.py
"""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "common"))

from GB_model import (  # noqa: E402
    SimpleVerticalLSTM,
    UnifiedAllSkyRTM_BiLSTM,
    UnifiedAllSkyRTM_MLP,
    UnifiedAllSkyRTM_Transformer,
)
from manuscript_config import MODEL_CONFIGS  # noqa: E402
from revised_model import build_model  # noqa: E402

torch.manual_seed(0)
N_CHANNELS = 22

models = {
    "transformer": UnifiedAllSkyRTM_Transformer(),
    "bilstm": UnifiedAllSkyRTM_BiLSTM(),
    "mlp": UnifiedAllSkyRTM_MLP(),
    "simple_lstm": SimpleVerticalLSTM(),
}

for name, model in models.items():
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = MODEL_CONFIGS[name]["parameters"]
    assert parameters == expected, (name, parameters, expected)
    print(f"{name:12s} params={parameters:8d}")

# Confirm support for both the original 37-level grid and the 137-level
# matched-resolution experiment.
for levels in (37, 137):
    batch = 2
    atm = torch.randn(batch, levels, 3)
    sfc = torch.randn(batch, 5)
    hydro = torch.rand(batch, levels - 1, 4)
    for name, model in models.items():
        candidate = build_model(
            name,
            n_layers=levels,
            n_channels=N_CHANNELS,
            hidden_dim=16,
            num_layers=1,
            nhead=4,
            num_transformer_layers=1,
            num_lstm_layers=1,
        ).eval()
        with torch.no_grad():
            y_clear = candidate(atm, sfc)
            y_cloudy = candidate(atm, sfc, hydro)
        assert y_clear.shape == (batch, N_CHANNELS), (name, levels, y_clear.shape)
        assert y_cloudy.shape == (batch, N_CHANNELS), (name, levels, y_cloudy.shape)
    print(f"all models forward OK at {levels} levels")

assert models["simple_lstm"].lstm.bidirectional is False

# Verify reverse-mode automatic differentiation through the recurrent model.
bilstm = UnifiedAllSkyRTM_BiLSTM(
    num_layers=37, num_channels=N_CHANNELS, hidden_dim=16, num_bi_lstm_blocks=1
).train()
a = torch.randn(1, 37, 3, requires_grad=True)
y = bilstm(a, torch.randn(1, 5), torch.rand(1, 36, 4))
grad = torch.autograd.grad(y[0, 0], a)[0]
assert grad.shape == a.shape
print("BiLSTM autograd.grad OK", tuple(grad.shape))
print("ALL_SMOKE_TESTS_PASSED")
