from pathlib import Path
import sys
_QJRMS_ROOT = Path(__file__).resolve().parents[2]
_QJRMS_COMMON = _QJRMS_ROOT / 'experiments' / 'common'
_QJRMS_CORE = _QJRMS_COMMON  # Backward-compatible alias.
if str(_QJRMS_ROOT) not in sys.path:
    sys.path.insert(0, str(_QJRMS_ROOT))
if str(_QJRMS_COMMON) not in sys.path:
    sys.path.insert(0, str(_QJRMS_COMMON))

"""Train architecture-comparison models on a reviewer-revision HDF5 dataset."""

import argparse
import json
from pathlib import Path

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch
from torch.utils.data import DataLoader

from data_utils import ReviewerH5Dataset, create_demo_h5, infer_dataset_metadata
from manuscript_config import MODEL_CONFIGS, PRETRAIN_PROTOCOL
from revised_model import build_model, count_parameters
from training_utils import batch_jacobian_loss, forward_model, resolve_device, set_seed

MODEL_NAMES = ["transformer", "bilstm", "mlp", "simple_lstm"]


def model_kwargs(name, meta, hidden_dim=None, num_layers=None, nhead=None):
    key = name.lower().replace("original_", "")
    config = MODEL_CONFIGS.get(key)
    if config is None:
        raise ValueError("Unknown model: {}".format(name))
    kwargs = {
        "n_layers": meta["n_layers"],
        "n_channels": meta["n_channels"],
        "atm_dim": meta["atm_dim"],
        "surf_dim": meta["surf_dim"],
        "hydro_dim": meta["hydro_dim"],
        "hidden_dim": hidden_dim or config["hidden_dim"],
    }
    if key == "transformer":
        kwargs.update({
            "nhead": nhead or config["nhead"],
            "num_transformer_layers": num_layers or config["num_transformer_layers"],
        })
    elif key in {"bilstm", "mlp", "simple_lstm"}:
        kwargs["num_layers"] = num_layers or config["num_layers"]
    return kwargs


def run_epoch(model, loader, optimizer, device, jac_weight, jac_freq, jac_max_samples):
    model.train()
    task_total, total, count = 0.0, 0.0, 0
    for step, batch in enumerate(loader):
        batch = {key: value.to(device) for key, value in batch.items()}
        prediction = forward_model(model, batch)
        task_loss = torch.nn.functional.mse_loss(prediction, batch["output"])
        loss = task_loss
        if jac_weight > 0.0 and step % jac_freq == 0:
            loss = loss + jac_weight * batch_jacobian_loss(
                model, batch, device, max_samples=jac_max_samples
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        task_total += float(task_loss.detach().cpu())
        total += float(loss.detach().cpu())
        count += 1
    return {"task_loss": task_total / max(count, 1),
            "total_loss": total / max(count, 1)}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    squared_error, absolute_error, count = 0.0, 0.0, 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        error = forward_model(model, batch) - batch["output"]
        squared_error += float((error ** 2).sum().cpu())
        absolute_error += float(error.abs().sum().cpu())
        count += error.numel()
    return {"mse": squared_error / max(count, 1),
            "rmse": (squared_error / max(count, 1)) ** 0.5,
            "mae": absolute_error / max(count, 1)}


def train_one_model(name, data_path, out_dir, epochs=PRETRAIN_PROTOCOL["max_epochs"],
                    batch_size=PRETRAIN_PROTOCOL["train_batch_size"],
                    max_train=None, max_val=None, hidden_dim=None, num_layers=None,
                    nhead=None, learning_rate=PRETRAIN_PROTOCOL["learning_rate"],
                    weight_decay=PRETRAIN_PROTOCOL["weight_decay"], jac_weight=0.0,
                    jac_freq=1, jac_max_samples=2, device="cpu", num_workers=8,
                    seed=2026, early_stop_patience=PRETRAIN_PROTOCOL["early_stop_patience"],
                    valid_batch_size=PRETRAIN_PROTOCOL["valid_batch_size"]):
    set_seed(seed)
    meta = infer_dataset_metadata(data_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device)
    train_ds = ReviewerH5Dataset(data_path, "train", "both", max_samples=max_train, seed=seed)
    valid_ds = ReviewerH5Dataset(data_path, "valid", "both", max_samples=max_val, seed=seed + 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"))
    valid_loader = DataLoader(valid_ds, batch_size=valid_batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"))
    kwargs = model_kwargs(name, meta, hidden_dim, num_layers, nhead)
    model = build_model(name, **kwargs).to(device)
    expected_parameters = MODEL_CONFIGS[name.lower().replace("original_", "")]["parameters"]
    using_reported_config = hidden_dim is None and num_layers is None and nhead is None
    if using_reported_config and meta["n_layers"] == 37 and meta["n_channels"] == 22:
        actual_parameters = count_parameters(model)
        if actual_parameters != expected_parameters:
            raise ValueError(
                "{} has {} parameters; manuscript reports {}".format(
                    name, actual_parameters, expected_parameters
                )
            )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    history = []
    best = {"rmse": float("inf"), "epoch": -1}
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, device, jac_weight,
                                jac_freq, jac_max_samples)
        valid_stats = validate(model, valid_loader, device)
        history.append({"epoch": epoch, **train_stats,
                        **{"val_" + key: value for key, value in valid_stats.items()}})
        scheduler.step(valid_stats["rmse"])
        print("[{}] epoch {} train_task={:.6e} val_rmse={:.6e}".format(
            name, epoch, train_stats["task_loss"], valid_stats["rmse"]))
        if valid_stats["rmse"] < best["rmse"]:
            stale_epochs = 0
            best = {"rmse": valid_stats["rmse"], "epoch": epoch}
            checkpoint = {
                "model_name": name, "model_kwargs": kwargs,
                "state_dict": model.state_dict(), "metadata": meta,
                "training_args": {
                    "epochs": epochs, "batch_size": batch_size, "max_train": max_train,
                    "max_val": max_val, "hidden_dim": hidden_dim, "num_layers": num_layers,
                    "nhead": nhead, "learning_rate": learning_rate,
                    "weight_decay": weight_decay, "jac_weight": jac_weight,
                    "jac_freq": jac_freq, "jac_max_samples": jac_max_samples,
                    "seed": seed,
                },
                "best_validation": best, "history": history,
            }
            torch.save(checkpoint, out_dir / (name + ".pth"))
        else:
            stale_epochs += 1
            if stale_epochs >= early_stop_patience:
                print("[{}] early stopping at epoch {}".format(name, epoch))
                break
    summary = {"model": name, "parameters": count_parameters(model),
               "best_val_rmse": best["rmse"], "best_epoch": best["epoch"],
               "data": str(data_path), "n_layers": meta["n_layers"],
               "n_channels": meta["n_channels"]}
    (out_dir / (name + "_summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / (name + "_history.json")).write_text(json.dumps(history, indent=2), encoding="utf-8")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--demo-dir", default="./reviewer_revision/results_demo")
    parser.add_argument("--models", default=",".join(MODEL_NAMES))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=PRETRAIN_PROTOCOL["max_epochs"])
    parser.add_argument("--batch-size", type=int, default=PRETRAIN_PROTOCOL["train_batch_size"])
    parser.add_argument("--val-batch-size", type=int, default=PRETRAIN_PROTOCOL["valid_batch_size"])
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=PRETRAIN_PROTOCOL["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=PRETRAIN_PROTOCOL["weight_decay"])
    parser.add_argument("--early-stop-patience", type=int, default=PRETRAIN_PROTOCOL["early_stop_patience"])
    parser.add_argument("--jac-weight", type=float, default=0.0)
    parser.add_argument("--jac-freq", type=int, default=1)
    parser.add_argument("--jac-max-samples", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    return parser
def main():
    args = build_parser().parse_args()
    data_path = args.data
    if data_path is None and args.demo:
        data_path = str(Path(args.demo_dir) / "demo_37_to_37.h5")
        create_demo_h5(data_path, n_layers=37)
    if data_path is None:
        raise SystemExit("Provide --data or use --demo.")
    summaries = []
    for name in [item.strip() for item in args.models.split(",") if item.strip()]:
        summaries.append(train_one_model(
            name=name, data_path=data_path, out_dir=args.out_dir, epochs=args.epochs,
            batch_size=args.batch_size, max_train=args.max_train, max_val=args.max_val,
            hidden_dim=args.hidden_dim, num_layers=args.num_layers, nhead=args.nhead,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            jac_weight=args.jac_weight, jac_freq=args.jac_freq,
            jac_max_samples=args.jac_max_samples, device=args.device,
            num_workers=args.num_workers, seed=args.seed,
            early_stop_patience=args.early_stop_patience,
            valid_batch_size=args.val_batch_size,
        ))
    (Path(args.out_dir) / "training_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
