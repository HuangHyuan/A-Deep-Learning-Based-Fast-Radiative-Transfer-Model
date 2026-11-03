from pathlib import Path
import sys
_QJRMS_ROOT = Path(__file__).resolve().parents[2]
_QJRMS_COMMON = _QJRMS_ROOT / 'experiments' / 'common'
_QJRMS_CORE = _QJRMS_COMMON  # Backward-compatible alias.
if str(_QJRMS_ROOT) not in sys.path:
    sys.path.insert(0, str(_QJRMS_ROOT))
if str(_QJRMS_COMMON) not in sys.path:
    sys.path.insert(0, str(_QJRMS_COMMON))

"""Select 50k clear and 50k cloudy original 37-level Jacobian samples.

The source is the original radiation_data_jacobian.h5. June 16 is split 90/10
into train/validation, and December 16 is retained as the test month.
"""

import argparse
from pathlib import Path

import h5py
import joblib
import numpy as np


def scale_jacobian(array, kind, info):
    y_range = np.maximum(info["y"]["max"] - info["y"]["min"], 1e-12)
    if kind == "atm":
        array = np.transpose(array, (0, 2, 1, 3))
        x_range = np.maximum(
            info["atm"]["max_array"] - info["atm"]["min_array"], 1e-12
        )[:, :2]
        factor = y_range[None, None, :, None] / x_range[None, :, None, :]
    elif kind == "sfc":
        x_range = np.maximum(
            info["sfc"]["max"] - info["sfc"]["min"], 1e-12
        )[:3]
        factor = y_range[None, :, None] / x_range[None, None, :]
    else:
        array = np.transpose(array, (0, 2, 1, 3))
        x_range = np.maximum(
            info["hydro"]["max_array"] - info["hydro"]["min_array"], 1e-12
        )
        factor = y_range[None, None, :, None] / x_range[None, :, None, :]
    return (array * factor).astype(np.float32)


def write_group(root, label, source, indices, info):
    group = root.create_group(label)
    for key in ("atm", "sfc", "out"):
        group.create_dataset(
            key, data=source[key][indices], compression="gzip"
        )
    group.create_dataset(
        "jac_atm",
        data=scale_jacobian(source["jac_atm"][indices], "atm", info),
        compression="gzip",
    )
    group.create_dataset(
        "jac_sfc",
        data=scale_jacobian(source["jac_sfc"][indices], "sfc", info),
        compression="gzip",
    )
    if label == "cloudy":
        group.create_dataset(
            "hydro", data=source["hydro"][indices], compression="gzip"
        )
        group.create_dataset(
            "jac_cloud",
            data=scale_jacobian(source["jac_cloud"][indices], "cloud", info),
            compression="gzip",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--scalers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    info = joblib.load(args.scalers)
    n_month = 25000
    train_count = int(0.9 * n_month)
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(n_month)
    train_idx = np.sort(5 * n_month + permutation[:train_count])
    valid_idx = np.sort(5 * n_month + permutation[train_count:])
    test_idx = np.arange(11 * n_month, 12 * n_month)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.input, "r") as src, h5py.File(out_path, "w") as dst:
        dst.attrs["description"] = (
            "Original 37-level Jacobian subset: June/December 2023"
        )
        dst.attrs["jacobians_scaled"] = 1
        for split_name, indices in (
            ("train", train_idx),
            ("valid", valid_idx),
            ("test", test_idx),
        ):
            root = dst.create_group(split_name)
            for label in ("clear", "cloudy"):
                write_group(root, label, src["train"][label], indices, info)
    joblib.dump(info, out_path.parent / "scaler_info.pkl")
    print("saved", out_path)
    print(
        "train/valid per scene",
        train_count,
        len(valid_idx),
        "test per scene",
        len(test_idx),
    )


if __name__ == "__main__":
    main()
