from pathlib import Path
import sys
_QJRMS_ROOT = Path(__file__).resolve().parents[2]
_QJRMS_COMMON = _QJRMS_ROOT / 'experiments' / 'common'
_QJRMS_CORE = _QJRMS_COMMON  # Backward-compatible alias.
if str(_QJRMS_ROOT) not in sys.path:
    sys.path.insert(0, str(_QJRMS_ROOT))
if str(_QJRMS_COMMON) not in sys.path:
    sys.path.insert(0, str(_QJRMS_COMMON))

"""Train an original GB_model architecture with the original GB_FW_Train loop."""

import argparse
import importlib
import json
import os
import numpy as np
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

from revised_model import build_model, count_parameters


class SafeDataLoader(torch.utils.data.DataLoader):
    """Disable worker collate path for large multidimensional arrays."""

    def __init__(self, *args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs.pop("persistent_workers", None)
        kwargs.pop("prefetch_factor", None)
        super().__init__(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="original_transformer")
    parser.add_argument("--n-layers", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    project_root = str(_QJRMS_CORE)
    sys.path.insert(0, project_root)
    base = importlib.import_module("GB_FW_Train")
    base.DataLoader = SafeDataLoader

    def getitem_fixed(self, idx):
        if idx < self.clear_len:
            i = idx
            return (
                self.atm_clear[i],
                self.sfc_clear[i],
                np.zeros((self.atm_clear.shape[1] - 1, 4), dtype=np.float32),
                self.out_clear[i],
                0,
            )
        i = idx - self.clear_len
        return (
            self.atm_cloudy[i],
            self.sfc_cloudy[i],
            self.hydro_cloudy[i],
            self.out_cloudy[i],
            1,
        )

    base.UnifiedRadiationDataset.__getitem__ = getitem_fixed
    data_path = args.data
    train_dataset = base.UnifiedRadiationDataset(data_path, mode="train", max_samples=2400000)
    valid_dataset = base.UnifiedRadiationDataset(data_path, mode="valid", max_samples=800000)
    print("train/valid", len(train_dataset), len(valid_dataset))
    model = build_model(
        args.model,
        n_layers=args.n_layers,
        n_channels=22,
        atm_dim=3,
        surf_dim=5,
        hydro_dim=4,
    )
    print("parameters", count_parameters(model))
    base.train_unified_model(model, data_path, args.out)
    Path(args.out + ".meta.json").write_text(json.dumps({
        "model_name": args.model,
        "model_kwargs": {
            "n_layers": args.n_layers,
            "n_channels": 22,
            "atm_dim": 3,
            "surf_dim": 5,
            "hydro_dim": 4,
        },
        "parameters": count_parameters(model),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
