from pathlib import Path
import sys
_QJRMS_ROOT = Path(__file__).resolve().parents[2]
_QJRMS_COMMON = _QJRMS_ROOT / 'experiments' / 'common'
_QJRMS_CORE = _QJRMS_COMMON  # Backward-compatible alias.
if str(_QJRMS_ROOT) not in sys.path:
    sys.path.insert(0, str(_QJRMS_ROOT))
if str(_QJRMS_COMMON) not in sys.path:
    sys.path.insert(0, str(_QJRMS_COMMON))

"""Evaluate Transformer, BiLSTM and MLP Jacobians using the notebook conventions."""

import argparse
import csv
import gc
import importlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import h5py
import joblib
import numpy as np
import torch

PROJECT = str(_QJRMS_CORE)
sys.path.insert(0, PROJECT)
import GB_model as gb_model
from GB_model import UnifiedAllSkyRTM_BiLSTM, UnifiedAllSkyRTM_MLP, UnifiedAllSkyRTM_Transformer
from benchmark_autograd_jacobian import compute_jacobian_autograd

if not hasattr(gb_model, "UnifiedAllSkyRTM"):
    gb_model.UnifiedAllSkyRTM = gb_model.UnifiedAllSkyRTM_Transformer
tj = importlib.import_module("GB_J_Train")
reverse_jacobian_batch = tj.reverse_jacobian_batch
reverse_jacobian_sfc = tj.reverse_jacobian_sfc


def tensor_scalers(info, device):
    result = {}
    for key, value in info.items():
        if isinstance(value, dict):
            result[key] = {}
            for subkey, subvalue in value.items():
                result[key][subkey] = torch.as_tensor(subvalue, dtype=torch.float32, device=device)
        else:
            result[key] = value
    return result


def metric_dict(truth, pred):
    truth = np.asarray(truth, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    error = pred - truth
    if len(truth) == 0:
        return {"n": 0, "bias": np.nan, "mae": np.nan, "rmse": np.nan, "corr": np.nan, "cosine": np.nan}
    std_t, std_p = np.std(truth), np.std(pred)
    corr = np.corrcoef(truth, pred)[0, 1] if len(truth) > 1 and std_t > 1e-12 and std_p > 1e-12 else np.nan
    denom = np.linalg.norm(truth) * np.linalg.norm(pred)
    cosine = np.dot(truth, pred) / denom if denom > 1e-12 else np.nan
    return {"n": int(len(truth)), "bias": float(np.mean(error)), "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))), "corr": float(corr), "cosine": float(cosine)}


def append_metrics(rows, model_name, scene, variable, truth, pred):
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    base = {"model": model_name, "scene": scene, "variable": variable}
    rows.append({**base, "channel": "all", **metric_dict(truth, pred)})
    if truth.ndim == 3:
        for channel in range(truth.shape[1]):
            rows.append({**base, "channel": channel,
                         **metric_dict(truth[:, channel, :], pred[:, channel, :])})
    elif truth.ndim == 2:
        for channel in range(truth.shape[1]):
            rows.append({**base, "channel": channel,
                         **metric_dict(truth[:, channel], pred[:, channel])})


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return model


def model_specs():
    return {
        "transformer": {
            "class": UnifiedAllSkyRTM_Transformer(hidden_dim=160, nhead=4, num_transformer_layers=3),
            "checkpoint": str(Path(os.environ.get('QJRMS_MODEL_DIR', _QJRMS_ROOT / 'model_assets')) / 'unified_model_jacobian_finetuned.pth'),
            "method": "vmap",
        },
        "bilstm": {
            "class": UnifiedAllSkyRTM_BiLSTM(hidden_dim=180, num_bi_lstm_blocks=2),
            "checkpoint": str(Path(os.environ.get('QJRMS_RUNS_DIR', _QJRMS_ROOT / 'runs')) / 'jacobian_two_autograd_strict' / 'bilstm.pth'),
            "method": "autograd",
        },
        "mlp": {
            "class": UnifiedAllSkyRTM_MLP(hidden_dim=12, num_mlp_blocks=3),
            "checkpoint": str(Path(os.environ.get('QJRMS_RUNS_DIR', _QJRMS_ROOT / 'runs')) / 'jacobian_two_autograd_strict' / 'mlp.pth'),
            "method": "vmap",
        },
    }


def jacobian_batch(model, method, atm, sfc, hydro=None):
    if method == "autograd":
        inputs = {"atm": atm, "sfc": sfc}
        if hydro is not None:
            inputs["hydro"] = hydro
        jac = compute_jacobian_autograd(model, inputs, is_create_graph=False)
        if hydro is None:
            hydro_jac = torch.zeros((atm.shape[0], 22, atm.shape[1] - 1, 4),
                                    device=atm.device, dtype=atm.dtype)
        else:
            hydro_jac = jac["hydro"]
        return jac["atm"], jac["sfc"], hydro_jac
    jac_atm, jac_sfc, jac_hydro = tj.compute_model_jacobian(
        model, atm, sfc, hydro, batch_size=64, device=str(atm.device)
    )
    if hydro is None and jac_hydro.numel() == 0:
        jac_hydro = torch.zeros((atm.shape[0], 22, atm.shape[1] - 1, 4),
                                device=atm.device, dtype=atm.dtype)
    return jac_atm, jac_sfc, jac_hydro


def physical_jacobians(model, method, atm, sfc, hydro, scaler_tensors):
    jac_atm, jac_sfc, jac_hydro = jacobian_batch(model, method, atm, sfc, hydro)
    jac_atm = reverse_jacobian_batch(jac_atm, scaler_tensors["atm"], scaler_tensors["y"])
    jac_sfc = reverse_jacobian_sfc(jac_sfc, scaler_tensors["sfc"], scaler_tensors["y"])
    if hydro is not None:
        jac_hydro = reverse_jacobian_batch(jac_hydro, scaler_tensors["hydro"], scaler_tensors["y"])
    return (jac_atm.detach().cpu().numpy(), jac_sfc.detach().cpu().numpy(),
            jac_hydro.detach().cpu().numpy())


def inverse_y(values, info):
    values = np.asarray(values)
    return values * (np.asarray(info["max"]) - np.asarray(info["min"])) + np.asarray(info["min"])


def evaluate_model(name, spec, scene_data, scaler_info, scaler_tensors, device, batch_size):
    model = load_checkpoint(spec["class"], spec["checkpoint"]).to(device).eval()
    method = spec["method"]
    output = {}
    for scene, data in scene_data.items():
        n = len(data["atm"])
        preds = []
        jac_atm_all, jac_sfc_all, jac_hydro_all = [], [], []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            atm = torch.as_tensor(data["atm"][start:end], dtype=torch.float32, device=device)
            sfc = torch.as_tensor(data["sfc"][start:end], dtype=torch.float32, device=device)
            hydro = None
            if scene == "cloudy":
                hydro = torch.as_tensor(data["hydro"][start:end], dtype=torch.float32, device=device)
            with torch.no_grad():
                pred = model(atm, sfc, hydro) if hydro is not None else model(atm, sfc)
            preds.append(pred.detach().cpu().numpy())
            ja, js, jh = physical_jacobians(model, method, atm, sfc, hydro, scaler_tensors)
            jac_atm_all.append(ja)
            jac_sfc_all.append(js)
            if jh is not None and jh.size:
                jac_hydro_all.append(jh)
        output[scene] = {
            "pred": np.concatenate(preds, axis=0),
            "jac_atm": np.concatenate(jac_atm_all, axis=0),
            "jac_sfc": np.concatenate(jac_sfc_all, axis=0),
            "jac_hydro": np.concatenate(jac_hydro_all, axis=0) if jac_hydro_all else None,
        }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return output



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", default=str(Path(os.environ.get('QJRMS_DATA_DIR', _QJRMS_ROOT / 'data')) / 'radiation_data_jacobian.h5'))
    parser.add_argument("--scalers", default=str(Path(os.environ.get('QJRMS_MODEL_DIR', _QJRMS_ROOT / 'model_assets')) / 'scaler_info.pkl'))
    parser.add_argument("--out-dir", default=str(Path(os.environ.get('QJRMS_RUNS_DIR', _QJRMS_ROOT / 'runs')) / 'jacobian_three_eval'))
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.enabled = False
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except AttributeError:
        pass

    scaler_info = joblib.load(args.scalers)
    scaler_tensors = tensor_scalers(scaler_info, device)
    rng = np.random.default_rng(args.seed)
    scene_data = {}
    with h5py.File(args.h5, "r") as h5:
        for scene in ("clear", "cloudy"):
            group = h5["valid"][scene]
            n = len(group["atm"])
            count = min(args.max_samples, n)
            indices = np.sort(rng.choice(n, count, replace=False))
            data = {"indices": indices, "atm": group["atm"][indices], "sfc": group["sfc"][indices],
                    "out": group["out"][indices], "jac_atm": group["jac_atm"][indices],
                    "jac_sfc": group["jac_sfc"][indices]}
            if scene == "cloudy":
                data["hydro"] = group["hydro"][indices]
                data["jac_cloud"] = group["jac_cloud"][indices]
            scene_data[scene] = data
    print("selected", {k: len(v["atm"]) for k, v in scene_data.items()}, flush=True)

    rows = []
    for model_index, (name, spec) in enumerate(model_specs().items()):
        print("evaluating", name, flush=True)
        batch_size = 128 if name != "bilstm" else 256
        result = evaluate_model(name, spec, scene_data, scaler_info, scaler_tensors, device, batch_size)
        for scene, data in scene_data.items():
            pred_physical = inverse_y(result[scene]["pred"], scaler_info["y"])
            truth_physical = inverse_y(data["out"], scaler_info["y"])
            append_metrics(rows, name, scene, "task_bt", truth_physical, pred_physical)
            q = data["atm"][:, :, 0] * (scaler_info["atm"]["max_array"][:, 0] - scaler_info["atm"]["min_array"][:, 0]) + scaler_info["atm"]["min_array"][:, 0]
            q_eff_truth = data["jac_atm"][:, :, :, 0] * q[:, None, :]
            q_eff_pred = result[scene]["jac_atm"][:, :, :, 0] * q[:, None, :]
            append_metrics(rows, name, scene, "atm_q", q_eff_truth, q_eff_pred)
            append_metrics(rows, name, scene, "atm_t", data["jac_atm"][:, :, :, 1], result[scene]["jac_atm"][:, :, :, 1])
            append_metrics(rows, name, scene, "surface", data["jac_sfc"][:, :, :3], result[scene]["jac_sfc"][:, :, :3])
            if scene == "cloudy":
                append_metrics(rows, name, scene, "hydrometeor_2_CRWC", data["jac_cloud"][:, :, :, 2], result[scene]["jac_hydro"][:, :, :, 2])
            else:
                pass
        print("finished", name, flush=True)

    with (out_dir / "jacobian_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "settings.json").write_text(json.dumps({
        "h5": args.h5,
        "max_samples_per_scene": args.max_samples,
        "seed": args.seed,
        "models": {
            "transformer": "unified_model_jacobian_finetuned.pth",
            "bilstm": "jacobian_two_autograd_strict/bilstm.pth",
            "mlp": "jacobian_two_autograd_strict/mlp.pth",
        },
    }, indent=2), encoding="utf-8")
    print("saved", out_dir / "jacobian_metrics.csv", flush=True)


if __name__ == "__main__":
    main()
