from pathlib import Path
import sys
_QJRMS_ROOT = Path(__file__).resolve().parents[2]
_QJRMS_COMMON = _QJRMS_ROOT / 'experiments' / 'common'
_QJRMS_CORE = _QJRMS_COMMON  # Backward-compatible alias.
if str(_QJRMS_ROOT) not in sys.path:
    sys.path.insert(0, str(_QJRMS_ROOT))
if str(_QJRMS_COMMON) not in sys.path:
    sys.path.insert(0, str(_QJRMS_COMMON))

"""Build revision BT HDF5 files from paired ERA5 NetCDF and FWD txt files."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from prepare_revision_h5 import (
    fit_scalers,
    load_single_data,
    scale_jacobians,
    transform_data,
)
import h5py


def parse_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def load_date(dataset_dir, output_dir, scene, date, derive_column_water):
    nc_path = dataset_dir / "ifs_{}_{}.nc".format(scene, date)
    out_date = date[:7].replace("-", "_") + "-" + date[8:]
    txt_path = output_dir / "{}_{}.txt".format(scene, out_date)
    if not nc_path.exists() or not txt_path.exists():
        raise FileNotFoundError("Missing pair:\n{}\n{}".format(nc_path, txt_path))
    atm, sfc, hydro, out = load_single_data(
        txt_path, nc_path, derive_column_water=derive_column_water
    )
    channels, levels = out.shape[1], atm.shape[1]
    jac = {
        "jac_atm": np.zeros((len(atm), levels, channels, 2), dtype=np.float32),
        "jac_sfc": np.zeros((len(atm), channels, 3), dtype=np.float32),
        "jac_cloud": np.zeros(
            (len(atm), hydro.shape[1], channels, 4), dtype=np.float32
        ),
    }
    return {"atm": atm, "sfc": sfc, "hydro": hydro, "out": out, **jac}


def merge(items):
    keys = items[0].keys()
    return {
        key: np.concatenate([item[key] for item in items], axis=0)
        for key in keys
    }


def load_split(dataset_dir, output_dir, scenes, dates, derive_column_water):
    return {
        scene: merge(
            [
                load_date(
                    dataset_dir,
                    output_dir,
                    scene,
                    date,
                    derive_column_water,
                )
                for date in dates
            ]
        )
        for scene in scenes
    }


def save_group(root, name, data, scalers):
    scaled = transform_data(data, scalers)
    jac = scale_jacobians(data, scalers)
    group = root.create_group(name)
    group.create_dataset("atm", data=scaled["atm"], compression="gzip")
    group.create_dataset("sfc", data=scaled["sfc"], compression="gzip")
    group.create_dataset("out", data=scaled["out"], compression="gzip")
    group.create_dataset("jac_atm", data=jac["jac_atm"], compression="gzip")
    group.create_dataset("jac_sfc", data=jac["jac_sfc"], compression="gzip")
    if name == "cloudy":
        group.create_dataset("hydro", data=scaled["hydro"], compression="gzip")
        group.create_dataset(
            "jac_cloud", data=jac["jac_cloud"], compression="gzip"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--level-name", required=True)
    parser.add_argument("--train-dates", required=True)
    parser.add_argument("--valid-dates", required=True)
    parser.add_argument("--test-dates", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    derive = args.level_name == "L137"
    splits = {
        "train": load_split(
            dataset_dir,
            output_dir,
            ["clear", "cloudy"],
            parse_list(args.train_dates),
            derive,
        ),
        "valid": load_split(
            dataset_dir,
            output_dir,
            ["clear", "cloudy"],
            parse_list(args.valid_dates),
            derive,
        ),
        "test": load_split(
            dataset_dir,
            output_dir,
            ["clear", "cloudy"],
            parse_list(args.test_dates),
            derive,
        ),
    }
    scalers = fit_scalers(splits["train"]["clear"], splits["train"]["cloudy"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        h5.attrs["description"] = "Revision BT dataset: " + args.level_name
        h5.attrs["jacobians_scaled"] = 1
        h5.attrs["n_layers"] = splits["train"]["clear"]["atm"].shape[1]
        for split_name, split in splits.items():
            root = h5.create_group(split_name)
            save_group(root, "clear", split["clear"], scalers)
            save_group(root, "cloudy", split["cloudy"], scalers)
    joblib.dump(scalers, out_path.parent / "scaler_info.pkl")
    metadata = {}
    for split_name, split in splits.items():
        metadata[split_name] = {
            "clear": int(len(split["clear"]["atm"])),
            "cloudy": int(len(split["cloudy"]["atm"])),
            "dates": parse_list(getattr(args, split_name + "_dates")),
        }
    metadata["n_layers"] = int(splits["train"]["clear"]["atm"].shape[1])
    metadata["n_channels"] = int(splits["train"]["clear"]["out"].shape[1])
    (out_path.parent / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
