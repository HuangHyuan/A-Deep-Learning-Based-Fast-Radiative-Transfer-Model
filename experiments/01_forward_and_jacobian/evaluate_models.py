from pathlib import Path
import sys
_QJRMS_ROOT = Path(__file__).resolve().parents[2]
_QJRMS_COMMON = _QJRMS_ROOT / 'experiments' / 'common'
_QJRMS_CORE = _QJRMS_COMMON  # Backward-compatible alias.
if str(_QJRMS_ROOT) not in sys.path:
    sys.path.insert(0, str(_QJRMS_ROOT))
if str(_QJRMS_COMMON) not in sys.path:
    sys.path.insert(0, str(_QJRMS_COMMON))

# placeholder
"""Evaluate task accuracy and analytical Jacobians for all trained models."""

import argparse
from pathlib import Path

import numpy as np
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch
from torch.utils.data import DataLoader

from data_utils import ReviewerH5Dataset
from metrics import jacobian_metrics, task_metrics, write_records
from revised_model import build_model, count_parameters
from training_utils import analytical_jacobians, forward_model, resolve_device


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = build_model(checkpoint["model_name"], **checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return checkpoint, model


@torch.no_grad()
def predict_loader(model, loader, device):
    truth, prediction, cloudy_flags = [], [], []
    scaler = getattr(loader.dataset, "output_scaler", None)
    if scaler is None:
        output_min = output_range = None
    elif hasattr(scaler, "data_min_"):
        output_min = torch.as_tensor(scaler.data_min_, dtype=torch.float32)
        output_range = torch.as_tensor(
            scaler.data_max_ - scaler.data_min_, dtype=torch.float32
        )
    else:
        output_min = torch.as_tensor(scaler["min"], dtype=torch.float32)
        output_range = torch.as_tensor(
            np.asarray(scaler["max"]) - np.asarray(scaler["min"]),
            dtype=torch.float32,
        )
    for batch in loader:
        device_batch = {key: value.to(device) for key, value in batch.items()}
        predicted = forward_model(model, device_batch).cpu()
        if scaler is not None:
            predicted = predicted * output_range + output_min
        prediction.append(predicted)
        truth.append(batch["output_physical"])
        cloudy_flags.append(batch["is_cloudy"])
    return (torch.cat(truth).numpy(), torch.cat(prediction).numpy(),
            torch.cat(cloudy_flags).numpy())


def evaluate_one(checkpoint_path, data_path, out_dir, split="valid", max_samples=None,
                 max_jac_samples=4, device="cpu", include_levels=False):
    device = resolve_device(device)
    checkpoint, model = load_checkpoint(checkpoint_path, device)
    name = checkpoint["model_name"]
    dataset = ReviewerH5Dataset(data_path, split, "both", max_samples=max_samples,
                                seed=checkpoint["training_args"]["seed"])
    loader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=False)
    truth, prediction, cloudy = predict_loader(model, loader, device)
    task_rows = task_metrics(name, split + "_all", truth, prediction)
    task_rows += task_metrics(name, split + "_clear", truth[cloudy < 0.5],
                              prediction[cloudy < 0.5])
    if (cloudy > 0.5).any():
        task_rows += task_metrics(name, split + "_cloudy", truth[cloudy > 0.5],
                                  prediction[cloudy > 0.5])
    jacobian_rows = []
    n_jac = min(max_jac_samples, len(dataset))
    if n_jac:
        indices = np.linspace(0, len(dataset) - 1, n_jac, dtype=int)
        clear_idx = [idx for idx in indices if dataset[idx]["is_cloudy"].item() < 0.5]
        cloudy_idx = [idx for idx in indices if dataset[idx]["is_cloudy"].item() > 0.5]
        if clear_idx:
            atm = torch.stack([dataset[idx]["atm_input"] for idx in clear_idx]).to(device)
            sfc = torch.stack([dataset[idx]["sfc_input"] for idx in clear_idx]).to(device)
            pred = analytical_jacobians(model, atm, sfc)
            gt_atm = torch.stack([dataset[idx]["jac_atm"] for idx in clear_idx]).numpy()
            gt_sfc = torch.stack([dataset[idx]["jac_sfc"] for idx in clear_idx]).numpy()
            jacobian_rows += jacobian_metrics(name, split + "_clear", "atmosphere",
                                              gt_atm, pred[0], include_levels)
            jacobian_rows += jacobian_metrics(name, split + "_clear", "surface",
                                              gt_sfc, pred[1])
        if cloudy_idx:
            atm = torch.stack([dataset[idx]["atm_input"] for idx in cloudy_idx]).to(device)
            sfc = torch.stack([dataset[idx]["sfc_input"] for idx in cloudy_idx]).to(device)
            hydro = torch.stack([dataset[idx]["hydro_input"] for idx in cloudy_idx]).to(device)
            pred = analytical_jacobians(model, atm, sfc, hydro)
            gt_atm = torch.stack([dataset[idx]["jac_atm"] for idx in cloudy_idx]).numpy()
            gt_sfc = torch.stack([dataset[idx]["jac_sfc"] for idx in cloudy_idx]).numpy()
            gt_cloud = torch.stack([dataset[idx]["jac_cloud"] for idx in cloudy_idx]).numpy()
            jacobian_rows += jacobian_metrics(name, split + "_cloudy", "atmosphere",
                                              gt_atm, pred[0], include_levels)
            jacobian_rows += jacobian_metrics(name, split + "_cloudy", "surface",
                                              gt_sfc, pred[1])
            jacobian_rows += jacobian_metrics(name, split + "_cloudy", "hydrometeor",
                                              gt_cloud, pred[2], include_levels)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / ("{}_predictions.npz".format(name)), truth=truth,
                        prediction=prediction, cloudy=cloudy)
    return task_rows, jacobian_rows, {"model": name, "parameters": count_parameters(model)}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-jac-samples", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-levels", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    model_paths = sorted(Path(args.models_dir).glob("*.pth"))
    if not model_paths:
        raise SystemExit("No .pth checkpoints found in {}".format(args.models_dir))
    all_task, all_jac, summaries = [], [], []
    for model_path in model_paths:
        task, jac, summary = evaluate_one(
            model_path, args.data, args.out_dir, split=args.split,
            max_samples=args.max_samples, max_jac_samples=args.max_jac_samples,
            device=args.device, include_levels=args.include_levels,
        )
        all_task.extend(task)
        all_jac.extend(jac)
        summaries.append(summary)
    out = Path(args.out_dir)
    write_records(out / "task_metrics.csv", all_task)
    write_records(out / "jacobian_metrics.csv", all_jac)
    write_records(out / "model_summary.csv", summaries)
    print("Saved task_metrics.csv and jacobian_metrics.csv to {}".format(out))


if __name__ == "__main__":
    main()
