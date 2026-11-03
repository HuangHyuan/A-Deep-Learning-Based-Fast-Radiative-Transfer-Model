"""HDF5 dataset helpers and synthetic smoke-test data."""

from pathlib import Path
import h5py
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset


def f32(x):
    return np.asarray(x, dtype=np.float32)


def optional(group, key, shape):
    return f32(group[key][:]) if key in group else np.zeros(shape, dtype=np.float32)


class ReviewerH5Dataset(Dataset):
    def __init__(self, h5_path, mode="train", data_type="both", max_samples=None,
                 seed=2026, scaler_info_path=None):
        self.h5_path = str(h5_path)
        self.mode = mode
        self.data_type = data_type
        rng = np.random.default_rng(seed)
        info_path = Path(scaler_info_path) if scaler_info_path else Path(self.h5_path).parent / "scaler_info.pkl"
        self.jacobian_scaler = joblib.load(str(info_path)) if info_path.exists() else None
        self.output_scaler = self.jacobian_scaler.get("y") if self.jacobian_scaler else None

        def read_dataset(group, key, indices, default_shape):
            if key not in group:
                return np.zeros((len(indices),) + default_shape[1:], dtype=np.float32)
            if indices is None:
                return f32(group[key][:])
            return f32(group[key][indices])

        with h5py.File(self.h5_path, "r") as h5:
            root = h5[mode]
            self.jacobians_scaled = bool(h5.attrs.get("jacobians_scaled", 0)) or self.jacobian_scaler is None
            clear_total = root["clear"]["atm"].shape[0] if data_type in {"clear", "both"} else 0
            cloudy_total = root["cloudy"]["atm"].shape[0] if data_type in {"cloudy", "both"} else 0
            total = clear_total + cloudy_total
            if max_samples is not None and max_samples < total:
                half = max_samples // 2
                n_clear = min(clear_total, half)
                n_cloudy = min(cloudy_total, max_samples - n_clear)
                remaining = max_samples - n_clear - n_cloudy
                if remaining > 0:
                    add_clear = min(remaining, clear_total - n_clear)
                    n_clear += add_clear
                    remaining -= add_clear
                if remaining > 0:
                    n_cloudy += min(remaining, cloudy_total - n_cloudy)
                clear_indices = np.arange(n_clear, dtype=int)
                cloudy_indices = np.arange(n_cloudy, dtype=int)
            else:
                clear_indices = np.arange(clear_total) if clear_total else np.array([], dtype=int)
                cloudy_indices = np.arange(cloudy_total) if cloudy_total else np.array([], dtype=int)

            self.clear_len = len(clear_indices)
            self.cloudy_len = len(cloudy_indices)
            if self.clear_len:
                group = root["clear"]
                self.clear_atm = read_dataset(group, "atm", clear_indices, (0, 37, 3))
                self.clear_sfc = read_dataset(group, "sfc", clear_indices, (0, 5))
                self.clear_out = read_dataset(group, "out", clear_indices, (0, 22))
                levels, channels = self.clear_atm.shape[1], self.clear_out.shape[1]
                self.clear_jac_atm = self._prepare_jacobian(
                    read_dataset(group, "jac_atm", clear_indices, (0, channels, levels, 2)),
                    "atm", levels, channels,
                )
                self.clear_jac_sfc = self._prepare_jacobian(
                    read_dataset(group, "jac_sfc", clear_indices, (0, channels, 3)),
                    "sfc", levels, channels,
                )
            if self.cloudy_len:
                group = root["cloudy"]
                self.cloudy_atm = read_dataset(group, "atm", cloudy_indices, (0, 37, 3))
                self.cloudy_sfc = read_dataset(group, "sfc", cloudy_indices, (0, 5))
                self.cloudy_hydro = read_dataset(group, "hydro", cloudy_indices, (0, 36, 4))
                self.cloudy_out = read_dataset(group, "out", cloudy_indices, (0, 22))
                levels, channels = self.cloudy_atm.shape[1], self.cloudy_out.shape[1]
                hydro_levels = self.cloudy_hydro.shape[1]
                self.cloudy_jac_atm = self._prepare_jacobian(
                    read_dataset(group, "jac_atm", cloudy_indices, (0, channels, levels, 2)),
                    "atm", levels, channels,
                )
                self.cloudy_jac_sfc = self._prepare_jacobian(
                    read_dataset(group, "jac_sfc", cloudy_indices, (0, channels, 3)),
                    "sfc", levels, channels,
                )
                self.cloudy_jac_cloud = self._prepare_jacobian(
                    read_dataset(group, "jac_cloud", cloudy_indices, (0, channels, hydro_levels, 4)),
                    "cloud", levels, channels,
                )
        self.total_len = self.clear_len + self.cloudy_len

    @staticmethod
    def _canonical_jacobian(array, kind, levels, channels):
        array = f32(array)
        if kind in {"atm", "cloud"} and array.ndim == 4:
            if array.shape[1] == channels and array.shape[2] in {levels, levels - 1}:
                array = np.transpose(array, (0, 2, 1, 3))
        elif kind == "sfc" and array.ndim == 3:
            if array.shape[1] != channels and array.shape[2] == channels:
                array = np.transpose(array, (0, 2, 1))
        return array

    def _scale_jacobian(self, array, kind):
        if self.jacobians_scaled or self.jacobian_scaler is None:
            return array
        info = self.jacobian_scaler
        y_range = np.maximum(info["y"]["max"] - info["y"]["min"], 1e-12)
        if kind == "atm":
            x_range = np.maximum(info["atm"]["max_array"] - info["atm"]["min_array"], 1e-12)[:, :2]
            factor = y_range[None, None, :, None] / x_range[None, :, None, :]
        elif kind == "sfc":
            x_range = np.maximum(info["sfc"]["max"] - info["sfc"]["min"], 1e-12)[:3]
            factor = y_range[None, :, None] / x_range[None, None, :]
        else:
            x_range = np.maximum(
                info["hydro"]["max_array"] - info["hydro"]["min_array"], 1e-12
            )
            factor = y_range[None, None, :, None] / x_range[None, :, None, :]
            if array.shape[1] != factor.shape[1]:
                array = array[:, :factor.shape[1]]
        return (array * factor).astype(np.float32)

    def _prepare_jacobian(self, array, kind, levels, channels):
        array = self._canonical_jacobian(array, kind, levels, channels)
        return self._scale_jacobian(array, kind)

    def _physical_output(self, output):
        scaler = self.output_scaler
        if scaler is None:
            return output
        if hasattr(scaler, "inverse_transform"):
            return scaler.inverse_transform(output[None, :])[0].astype(np.float32)
        scale = np.asarray(scaler["max"]) - np.asarray(scaler["min"])
        return (output * scale + np.asarray(scaler["min"])).astype(np.float32)

    def _subset(self, indices):
        clear_indices = indices[indices < self.clear_len]
        cloudy_indices = indices[indices >= self.clear_len] - self.clear_len
        if len(clear_indices):
            self.clear_atm = self.clear_atm[clear_indices]
            self.clear_sfc = self.clear_sfc[clear_indices]
            self.clear_out = self.clear_out[clear_indices]
            self.clear_jac_atm = self.clear_jac_atm[clear_indices]
            self.clear_jac_sfc = self.clear_jac_sfc[clear_indices]
            self.clear_len = len(clear_indices)
        else:
            self.clear_len = 0
        if len(cloudy_indices):
            self.cloudy_atm = self.cloudy_atm[cloudy_indices]
            self.cloudy_sfc = self.cloudy_sfc[cloudy_indices]
            self.cloudy_hydro = self.cloudy_hydro[cloudy_indices]
            self.cloudy_out = self.cloudy_out[cloudy_indices]
            self.cloudy_jac_atm = self.cloudy_jac_atm[cloudy_indices]
            self.cloudy_jac_sfc = self.cloudy_jac_sfc[cloudy_indices]
            self.cloudy_jac_cloud = self.cloudy_jac_cloud[cloudy_indices]
            self.cloudy_len = len(cloudy_indices)
        else:
            self.cloudy_len = 0
        self.total_len = self.clear_len + self.cloudy_len

    @property
    def n_layers(self):
        return int(self.clear_atm.shape[1] if self.clear_len else self.cloudy_atm.shape[1])

    @property
    def n_channels(self):
        return int(self.clear_out.shape[1] if self.clear_len else self.cloudy_out.shape[1])

    @property
    def atm_dim(self):
        return int(self.clear_atm.shape[2] if self.clear_len else self.cloudy_atm.shape[2])

    @property
    def surf_dim(self):
        return int(self.clear_sfc.shape[1] if self.clear_len else self.cloudy_sfc.shape[1])

    @property
    def hydro_dim(self):
        return int(self.cloudy_hydro.shape[2]) if self.cloudy_len else 4

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        cloudy = idx >= self.clear_len
        if cloudy:
            idx = idx - self.clear_len
            hydro = torch.from_numpy(self.cloudy_hydro[idx])
            cloud_jac = torch.from_numpy(self.cloudy_jac_cloud[idx])
            if hydro.shape[0] < self.n_layers:
                hydro = torch.nn.functional.pad(hydro, (0, 0, 0, self.n_layers - hydro.shape[0]))
            if cloud_jac.shape[0] < self.n_layers:
                cloud_jac = torch.nn.functional.pad(
                    cloud_jac, (0, 0, 0, 0, 0, self.n_layers - cloud_jac.shape[0])
                )
            hydro = hydro[:self.n_layers]
            cloud_jac = cloud_jac[:self.n_layers]
            return {
                "atm_input": torch.from_numpy(self.cloudy_atm[idx]),
                "sfc_input": torch.from_numpy(self.cloudy_sfc[idx]),
                "hydro_input": hydro,
                "output": torch.from_numpy(self.cloudy_out[idx]),
                "output_physical": torch.from_numpy(self._physical_output(self.cloudy_out[idx])),
                "jac_atm": torch.from_numpy(self.cloudy_jac_atm[idx]),
                "jac_sfc": torch.from_numpy(self.cloudy_jac_sfc[idx]),
                "jac_cloud": cloud_jac,
                "is_cloudy": torch.tensor(1.0),
            }
        return {
            "atm_input": torch.from_numpy(self.clear_atm[idx]),
            "sfc_input": torch.from_numpy(self.clear_sfc[idx]),
            "hydro_input": torch.zeros(self.n_layers, self.hydro_dim),
            "output": torch.from_numpy(self.clear_out[idx]),
            "output_physical": torch.from_numpy(self._physical_output(self.clear_out[idx])),
            "jac_atm": torch.from_numpy(self.clear_jac_atm[idx]),
            "jac_sfc": torch.from_numpy(self.clear_jac_sfc[idx]),
            "jac_cloud": torch.zeros(self.n_layers, self.n_channels, self.hydro_dim),
            "is_cloudy": torch.tensor(0.0),
        }


def infer_dataset_metadata(h5_path):
    with h5py.File(h5_path, "r") as h5:
        root = h5["train"]
        clear = root["clear"]
        cloudy = root["cloudy"]
        clear_atm = clear["atm"]
        cloudy_atm = cloudy["atm"]
        out = clear["out"] if "out" in clear else cloudy["out"]
        sfc = clear["sfc"] if "sfc" in clear else cloudy["sfc"]
        hydro = cloudy["hydro"]
        return {
            "n_layers": int(clear_atm.shape[1] if clear_atm.shape[0] else cloudy_atm.shape[1]),
            "n_channels": int(out.shape[1]),
            "atm_dim": int(clear_atm.shape[2] if clear_atm.shape[0] else cloudy_atm.shape[2]),
            "surf_dim": int(sfc.shape[1]),
            "hydro_dim": int(hydro.shape[2]),
        }


def _profile(rng, count, levels, amplitude):
    z = np.linspace(0.0, 1.0, levels, dtype=np.float32)
    value = amplitude * (np.sin(2.0 * np.pi * (z[None, :] + 0.15 * rng.random((count, 1))))
                         + 0.5 * np.cos(3.0 * np.pi * z[None, :]))
    value += rng.normal(0.0, 0.015, size=value.shape)
    return np.clip(value, 0.0, 1.0).astype(np.float32)


def _make_group(rng, count, levels, channels, cloudy):
    atm_q = _profile(rng, count, levels, 0.25) + 0.25
    atm_t = _profile(rng, count, levels, 0.20) + 0.55
    atm_p = np.repeat(np.linspace(0.05, 1.0, levels, dtype=np.float32)[None, :], count, axis=0)
    atm = np.stack([atm_q, atm_t, atm_p], axis=-1).astype(np.float32)
    sfc = rng.uniform(0.0, 1.0, size=(count, 5)).astype(np.float32)
    if cloudy:
        hydro = np.abs(_profile(rng, count, levels, 0.08))
        hydro = np.repeat(hydro[:, :, None], 4, axis=2)
        hydro *= rng.uniform(0.2, 1.0, size=(count, 1, 4)).astype(np.float32)
        hydro[:, :5, :] *= 0.2
        hydro[hydro < 0.002] = 0.0
    else:
        sfc[:, 3:] = 0.0
        hydro = np.zeros((count, levels, 4), dtype=np.float32)
    channel_scale = np.linspace(0.85, 1.15, channels, dtype=np.float32)[None, :]
    base = 0.15 + 0.42 * atm[:, :, 1].mean(1, keepdims=True)
    base += 0.18 * atm[:, :, 0].mean(1, keepdims=True) + 0.12 * sfc[:, 0:1]
    if cloudy:
        base += 0.09 * hydro.mean(axis=(1, 2))[:, None]
    out = np.clip(base * channel_scale + rng.normal(0.0, 0.005, size=(count, channels)), 0, 1).astype(np.float32)
    level = np.linspace(1.0, 0.5, levels, dtype=np.float32)[None, :, None, None]
    chan = np.linspace(0.8, 1.2, channels, dtype=np.float32)[None, None, :, None]
    jac_atm = np.zeros((count, levels, channels, 2), dtype=np.float32)
    jac_atm[:, :, :, 0] = (0.06 * level * chan)[0, :, :, 0]
    jac_atm[:, :, :, 1] = (0.18 * (1.0 - level) * chan)[0, :, :, 0]
    jac_atm += rng.normal(0.0, 0.002, size=jac_atm.shape).astype(np.float32)
    jac_sfc = rng.normal(0.0, 0.015, size=(count, channels, 3)).astype(np.float32)
    jac_sfc[:, :, 0] += 0.12
    jac_sfc[:, :, 1] += 0.05
    jac_cloud = np.zeros((count, levels, channels, 4), dtype=np.float32)
    if cloudy:
        hydro_scale = np.linspace(1.0, 0.3, levels, dtype=np.float32)[None, :, None, None]
        jac_cloud[:] = 0.025 * hydro_scale
        jac_cloud += rng.normal(0.0, 0.001, size=jac_cloud.shape).astype(np.float32)
    return atm, sfc, hydro, out, jac_atm, jac_sfc, jac_cloud


def create_demo_h5(path, n_layers=37, n_channels=22, n_train_clear=12,
                   n_train_cloudy=12, n_valid_clear=4, n_valid_cloudy=4, seed=2026):
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["description"] = "Synthetic smoke-test data; replace with physical data."
        h5.attrs["jacobians_scaled"] = 1
        h5.attrs["n_layers"] = int(n_layers)
        h5.attrs["n_channels"] = int(n_channels)
        h5.attrs["scaled"] = True
        for mode, n_clear, n_cloudy in (("train", n_train_clear, n_train_cloudy),
                                         ("valid", n_valid_clear, n_valid_cloudy)):
            root = h5.create_group(mode)
            for label, count, cloudy in (("clear", n_clear, False), ("cloudy", n_cloudy, True)):
                group = root.create_group(label)
                arrays = _make_group(rng, count, n_layers, n_channels, cloudy)
                atm, sfc, hydro, out, jac_atm, jac_sfc, jac_cloud = arrays
                group.create_dataset("atm", data=atm)
                group.create_dataset("sfc", data=sfc)
                if cloudy:
                    group.create_dataset("hydro", data=hydro)
                group.create_dataset("out", data=out)
                group.create_dataset("jac_atm", data=jac_atm)
                group.create_dataset("jac_sfc", data=jac_sfc)
                if cloudy:
                    group.create_dataset("jac_cloud", data=jac_cloud)
    return str(path)


def create_demo_suite(out_dir):
    out = Path(out_dir)
    return {
        "37_to_37": create_demo_h5(out / "demo_37_to_37.h5", n_layers=37, seed=37037),
        "137_to_137": create_demo_h5(out / "demo_137_to_137.h5", n_layers=137, seed=137137),
        "37_to_137": create_demo_h5(out / "demo_37_to_137.h5", n_layers=37, seed=37137),
    }


__all__ = ["ReviewerH5Dataset", "infer_dataset_metadata", "create_demo_h5", "create_demo_suite"]
