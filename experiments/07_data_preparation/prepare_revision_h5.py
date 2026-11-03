from pathlib import Path
import sys
_QJRMS_ROOT = Path(__file__).resolve().parents[2]
_QJRMS_COMMON = _QJRMS_ROOT / 'experiments' / 'common'
_QJRMS_CORE = _QJRMS_COMMON  # Backward-compatible alias.
if str(_QJRMS_ROOT) not in sys.path:
    sys.path.insert(0, str(_QJRMS_ROOT))
if str(_QJRMS_COMMON) not in sys.path:
    sys.path.insert(0, str(_QJRMS_COMMON))

"""Build a normalized revision HDF5 file from simulation, NetCDF and Jacobian files.

The output schema is identical to the original scripts, but all Jacobians are
stored in model coordinates (scaled output / scaled input). This makes
architecture comparisons independent of the original MinMaxScaler ranges.
"""

import argparse
import json
from pathlib import Path

from netCDF4 import Dataset
import h5py
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def profiles_to_samples(array):
    """Move the vertical/profile dimension to axis 1 for a 2-D array."""

    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError("Expected a 2-D profile variable, got {}".format(array.shape))
    first, second = array.shape
    if first in (37, 137) or first < second:
        return array.T
    if second in (37, 137):
        return array
    return array.T



def load_single_data(sim_path, nc_path, derive_column_water=False):
    output = np.loadtxt(sim_path)
    if output.ndim == 1:
        output = output[None, :]
    with Dataset(nc_path, "r") as nc:
        names = ["Level_H2O", "Level_O3", "Level_Pressure", "Level_Temperature",
                 "geopotential_height", "Level_CLWC", "Level_CIWC",
                 "Level_CRWC", "Level_CSWC"]
        profiles = {name: profiles_to_samples(nc.variables[name][:]) for name in names}
        surface = {
            "Temperature_2M": np.asarray(nc.variables["Temperature_2M"][:]).reshape(-1),
            "H2O_2M": np.asarray(nc.variables["H2O_2M"][:]).reshape(-1),
            "Surface_Pressure": np.asarray(nc.variables["Surface_Pressure"][:]).reshape(-1),
        }
        if derive_column_water:
            surface["TCLW"] = profiles["Level_CLWC"].sum(axis=1)
            surface["TCRW"] = profiles["Level_CRWC"].sum(axis=1)
        else:
            surface["TCLW"] = np.asarray(nc.variables["TCLW"][:]).reshape(-1)
            surface["TCRW"] = np.asarray(nc.variables["TCRW"][:]).reshape(-1)
    n_samples = profiles["Level_H2O"].shape[0]
    if output.shape[0] != n_samples and output.shape[1] == n_samples:
        output = output.T
    if output.shape[0] != n_samples:
        raise ValueError("Simulation/NetCDF sample mismatch")
    height = profiles["geopotential_height"]
    dz = height[:, :-1] - height[:, 1:]
    atm = np.stack([profiles["Level_H2O"], profiles["Level_Temperature"],
                    profiles["Level_Pressure"]], axis=-1).astype(np.float32)
    sfc = np.stack([surface[name] for name in
                    ["Temperature_2M", "H2O_2M", "Surface_Pressure", "TCLW", "TCRW"]],
                   axis=-1).astype(np.float32)
    hydro = []
    for name in ["Level_CLWC", "Level_CRWC", "Level_CSWC", "Level_CIWC"]:
        mid = 0.5 * (profiles[name][:, :-1] + profiles[name][:, 1:])
        mid[mid < 1e-8] = 0.0
        hydro.append(mid * dz)
    return atm, sfc, np.stack(hydro, axis=-1).astype(np.float32), output.astype(np.float32)


def orient_to(array, target_sizes):
    array = np.asarray(array)
    if array.ndim != len(target_sizes):
        raise ValueError("Jacobian rank mismatch: {} vs {}".format(array.shape, target_sizes))
    used, permutation = set(), []
    for size in target_sizes:
        candidates = [axis for axis, value in enumerate(array.shape)
                      if value == size and axis not in used]
        if not candidates:
            raise ValueError("Cannot orient {} to {}".format(array.shape, target_sizes))
        permutation.append(candidates[0])
        used.add(candidates[0])
    return np.transpose(array, permutation)


def load_jacobian(jac_path, n_samples, levels, channels, cloudy):
    with Dataset(jac_path, "r") as ds:
        raw_q = ds.variables["jacobian_h2o"][:]
        raw_t = ds.variables["jacobian_temperature"][:]
        raw_s = ds.variables["jacobian_surface"][:]
        raw_c = ds.variables["jacobian_cloud"][:] if cloudy and "jacobian_cloud" in ds.variables else None
    jac_q = orient_to(raw_q, (n_samples, levels, channels))
    jac_t = orient_to(raw_t, (n_samples, levels, channels))
    jac_atm = np.stack([jac_q, jac_t], axis=-1)
    jac_sfc = orient_to(raw_s, (n_samples, channels, 3))
    if cloudy and raw_c is not None:
        cloud_levels = levels - 1 if levels - 1 in raw_c.shape else levels
        jac_cloud = orient_to(raw_c, (n_samples, cloud_levels, channels, 4))
    else:
        jac_cloud = np.zeros((n_samples, levels, channels, 4), dtype=np.float32)
    return jac_atm.astype(np.float32), jac_sfc.astype(np.float32), jac_cloud


def load_directory(sim_dir, nc_dir, jac_dir=None, sim_glob="*.txt", nc_glob="*.nc",
                   jac_glob="*.nc", cloudy=False, limit=None):
    sim_files = sorted(Path(sim_dir).glob(sim_glob))
    nc_files = sorted(Path(nc_dir).glob(nc_glob))
    if limit is not None:
        sim_files, nc_files = sim_files[:limit], nc_files[:limit]
    if len(sim_files) != len(nc_files):
        raise ValueError("Simulation/NetCDF file counts differ: {} vs {}".format(
            len(sim_files), len(nc_files)))
    jac_files = sorted(Path(jac_dir).glob(jac_glob)) if jac_dir else [None] * len(sim_files)
    if len(jac_files) != len(sim_files):
        raise ValueError("Jacobian file count differs: {} vs {}".format(
            len(jac_files), len(sim_files)))
    all_atm, all_sfc, all_hydro, all_out = [], [], [], []
    all_jac_atm, all_jac_sfc, all_jac_cloud = [], [], []
    for sim_path, nc_path, jac_path in zip(sim_files, nc_files, jac_files):
        atm, sfc, hydro, out = load_single_data(sim_path, nc_path)
        all_atm.append(atm)
        all_sfc.append(sfc)
        all_hydro.append(hydro)
        all_out.append(out)
        if jac_path is None:
            n_samples, levels, channels = atm.shape[0], atm.shape[1], out.shape[1]
            all_jac_atm.append(np.zeros((n_samples, levels, channels, 2), dtype=np.float32))
            all_jac_sfc.append(np.zeros((n_samples, channels, 3), dtype=np.float32))
            all_jac_cloud.append(np.zeros((n_samples, levels, channels, 4), dtype=np.float32))
        else:
            jac = load_jacobian(jac_path, atm.shape[0], atm.shape[1], out.shape[1], cloudy)
            all_jac_atm.append(jac[0])
            all_jac_sfc.append(jac[1])
            all_jac_cloud.append(jac[2])
    return {
        "atm": np.concatenate(all_atm), "sfc": np.concatenate(all_sfc),
        "hydro": np.concatenate(all_hydro), "out": np.concatenate(all_out),
        "jac_atm": np.concatenate(all_jac_atm),
        "jac_sfc": np.concatenate(all_jac_sfc),
        "jac_cloud": np.concatenate(all_jac_cloud),
    }


def fit_scalers(clear, cloudy):
    atm_scalers, hydro_scalers = [], []
    for level in range(clear["atm"].shape[1]):
        atm_scalers.append(MinMaxScaler().fit(np.concatenate(
            [clear["atm"][:, level], cloudy["atm"][:, level]], axis=0)))
    for level in range(cloudy["hydro"].shape[1]):
        hydro_scalers.append(MinMaxScaler().fit(cloudy["hydro"][:, level]))
    sfc_scaler = MinMaxScaler().fit(np.concatenate([clear["sfc"], cloudy["sfc"]], axis=0))
    y_scaler = MinMaxScaler().fit(np.concatenate([clear["out"], cloudy["out"]], axis=0))
    return {"atm": atm_scalers, "sfc": sfc_scaler, "hydro": hydro_scalers, "y": y_scaler}


def transform_data(data, scalers):
    result = {
        "atm": np.zeros_like(data["atm"]),
        "sfc": scalers["sfc"].transform(data["sfc"]),
        "hydro": np.zeros_like(data["hydro"]),
        "out": scalers["y"].transform(data["out"]),
    }
    for level, scaler in enumerate(scalers["atm"]):
        result["atm"][:, level] = scaler.transform(data["atm"][:, level])
    for level, scaler in enumerate(scalers["hydro"]):
        result["hydro"][:, level] = scaler.transform(data["hydro"][:, level])
    return {key: value.astype(np.float32) for key, value in result.items()}


def scale_jacobians(data, scalers):
    atm_range = np.stack([s.data_max_ - s.data_min_ for s in scalers["atm"]])[:, :2]
    hydro_range = np.stack([s.data_max_ - s.data_min_ for s in scalers["hydro"]])
    sfc_range = (scalers["sfc"].data_max_ - scalers["sfc"].data_min_)[:3]
    y_range = scalers["y"].data_max_ - scalers["y"].data_min_
    atm_factor = y_range[None, None, :, None] / np.maximum(atm_range[None, :, None, :], 1e-12)
    sfc_factor = y_range[None, :, None] / np.maximum(sfc_range[None, None, :], 1e-12)
    hydro_factor = y_range[None, None, :, None] / np.maximum(hydro_range[None, :, None, :], 1e-12)
    cloud = data["jac_cloud"]
    if cloud.shape[1] != hydro_factor.shape[1]:
        cloud = cloud[:, :hydro_factor.shape[1]]
    return {
        "jac_atm": (data["jac_atm"] * atm_factor).astype(np.float32),
        "jac_sfc": (data["jac_sfc"] * sfc_factor).astype(np.float32),
        "jac_cloud": (cloud * hydro_factor).astype(np.float32),
    }


def scaler_info(scalers):
    return {
        "atm": {
            "min_array": np.stack([s.data_min_ for s in scalers["atm"]])[:, :2],
            "max_array": np.stack([s.data_max_ for s in scalers["atm"]])[:, :2],
        },
        "sfc": {"min": scalers["sfc"].data_min_, "max": scalers["sfc"].data_max_},
        "hydro": {
            "min_array": np.stack([s.data_min_ for s in scalers["hydro"]]),
            "max_array": np.stack([s.data_max_ for s in scalers["hydro"]]),
        },
        "y": {"min": scalers["y"].data_min_, "max": scalers["y"].data_max_},
    }


def save_split(h5_root, name, clear, clear_scaled, clear_jac, cloudy, cloudy_scaled, cloudy_jac):
    root = h5_root.create_group(name)
    group = root.create_group("clear")
    group.create_dataset("atm", data=clear_scaled["atm"], compression="gzip")
    group.create_dataset("sfc", data=clear_scaled["sfc"], compression="gzip")
    group.create_dataset("out", data=clear_scaled["out"], compression="gzip")
    group.create_dataset("jac_atm", data=clear_jac["jac_atm"], compression="gzip")
    group.create_dataset("jac_sfc", data=clear_jac["jac_sfc"], compression="gzip")
    group = root.create_group("cloudy")
    group.create_dataset("atm", data=cloudy_scaled["atm"], compression="gzip")
    group.create_dataset("sfc", data=cloudy_scaled["sfc"], compression="gzip")
    group.create_dataset("hydro", data=cloudy_scaled["hydro"], compression="gzip")
    group.create_dataset("out", data=cloudy_scaled["out"], compression="gzip")
    group.create_dataset("jac_atm", data=cloudy_jac["jac_atm"], compression="gzip")
    group.create_dataset("jac_sfc", data=cloudy_jac["jac_sfc"], compression="gzip")
    group.create_dataset("jac_cloud", data=cloudy_jac["jac_cloud"], compression="gzip")


def load_split(split_config):
    clear = load_directory(**split_config["clear"], cloudy=False)
    cloudy = load_directory(**split_config["cloudy"], cloudy=True)
    clear["sfc"][:, 3:] = 0.0
    return clear, cloudy


def run(config):
    train_clear, train_cloudy = load_split(config["train"])
    valid_clear, valid_cloudy = load_split(config["valid"])
    scalers = fit_scalers(train_clear, train_cloudy)
    output_h5 = Path(config["output_h5"])
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as h5:
        h5.attrs["description"] = "Reviewer-revision dataset; Jacobians stored in scaled coordinates."
        h5.attrs["jacobians_scaled"] = 1
        h5.attrs["n_layers"] = train_clear["atm"].shape[1]
        h5.attrs["n_channels"] = train_clear["out"].shape[1]
        save_split(h5, "train", train_clear, transform_data(train_clear, scalers),
                   scale_jacobians(train_clear, scalers), train_cloudy,
                   transform_data(train_cloudy, scalers), scale_jacobians(train_cloudy, scalers))
        save_split(h5, "valid", valid_clear, transform_data(valid_clear, scalers),
                   scale_jacobians(valid_clear, scalers), valid_cloudy,
                   transform_data(valid_cloudy, scalers), scale_jacobians(valid_cloudy, scalers))
    scaler_path = config.get("output_scalers", str(output_h5.with_suffix(".scalers.pkl")))
    joblib.dump(scaler_info(scalers), scaler_path)
    print("Saved {} and {}".format(output_h5, scaler_path))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON experiment configuration.")
    return parser


def main():
    args = build_parser().parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    run(config)


if __name__ == "__main__":
    main()
