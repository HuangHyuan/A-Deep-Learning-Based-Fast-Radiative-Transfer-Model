from pathlib import Path
import sys
_QJRMS_ROOT = Path(__file__).resolve().parents[2]
_QJRMS_COMMON = _QJRMS_ROOT / 'experiments' / 'common'
_QJRMS_CORE = _QJRMS_COMMON  # Backward-compatible alias.
if str(_QJRMS_ROOT) not in sys.path:
    sys.path.insert(0, str(_QJRMS_ROOT))
if str(_QJRMS_COMMON) not in sys.path:
    sys.path.insert(0, str(_QJRMS_COMMON))

"""Convert the original HDF5 Jacobians to scaled/model coordinates.

The original scripts store inputs and outputs in MinMaxScaler coordinates but
store Jacobians in physical units. The architecture comparison trains with a
Jacobian-consistency loss and therefore needs both sides in the same
coordinate system. This utility uses scaler_info.pkl to perform that conversion.
"""

import argparse
from pathlib import Path

import h5py
import joblib
import numpy as np


def scale_jacobians(group, info):
    atm_range = np.maximum(info["atm"]["max_array"] - info["atm"]["min_array"], 1e-12)[:, :2]
    sfc_range = np.maximum(info["sfc"]["max"] - info["sfc"]["min"], 1e-12)[:3]
    hydro_range = np.maximum(info["hydro"]["max_array"] - info["hydro"]["min_array"], 1e-12)
    y_range = np.maximum(info["y"]["max"] - info["y"]["min"], 1e-12)
    result = {}
    if "jac_atm" in group:
        factor = y_range[None, None, :, None] / atm_range[None, :, None, :]
        result["jac_atm"] = (group["jac_atm"][:] * factor).astype(np.float32)
    if "jac_sfc" in group:
        factor = y_range[None, :, None] / sfc_range[None, None, :]
        result["jac_sfc"] = (group["jac_sfc"][:] * factor).astype(np.float32)
    if "jac_cloud" in group:
        factor = y_range[None, None, :, None] / hydro_range[None, :, None, :]
        result["jac_cloud"] = (group["jac_cloud"][:] * factor).astype(np.float32)
    return result


def convert(input_h5, scaler_path, output_h5):
    info = joblib.load(scaler_path)
    with h5py.File(input_h5, "r") as src, h5py.File(output_h5, "w") as dst:
        dst.attrs["description"] = "Converted original HDF5; Jacobians in scaled coordinates."
        dst.attrs["jacobians_scaled"] = 1
        for split in ("train", "valid"):
            split_group = dst.create_group(split)
            for label in ("clear", "cloudy"):
                src_group = src[split][label]
                dst_group = split_group.create_group(label)
                for key in src_group:
                    if key.startswith("jac_"):
                        continue
                    dst_group.create_dataset(key, data=src_group[key][:], compression="gzip")
                for key, value in scale_jacobians(src_group, info).items():
                    dst_group.create_dataset(key, data=value, compression="gzip")
    print("Saved {}".format(output_h5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--scalers", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    convert(args.input, args.scalers, args.output)


if __name__ == "__main__":
    main()
