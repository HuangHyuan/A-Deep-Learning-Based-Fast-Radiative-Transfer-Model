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
"""Run the 37-to-37, 137-to-137 and 37-to-137 truncation experiments."""

import argparse
from pathlib import Path

from data_utils import create_demo_h5
from evaluate_models import evaluate_one
from metrics import write_records
from train_models import train_one_model


def run_one(label, data_path, out_dir, args):
    run_dir = Path(out_dir) / label
    summary = train_one_model(
        "transformer", data_path, run_dir, epochs=args.epochs,
        batch_size=args.batch_size, max_train=args.max_train, max_val=args.max_val,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers, nhead=args.nhead,
        learning_rate=args.learning_rate, jac_weight=0.0, device=args.device,
        num_workers=args.num_workers, seed=args.seed,
    )
    task_rows, _, _ = evaluate_one(
        run_dir / "transformer.pth", data_path, run_dir, split="valid",
        max_samples=args.max_val, max_jac_samples=0, device=args.device,
    )
    aggregate = next(row for row in task_rows if row["split"] == "valid_all"
                     and str(row["channel"]) == "all")
    return {
        "experiment": label,
        "input_levels": 137 if "137" in data_path else 37,
        "reference_levels": 137 if label != "matched_37" else 37,
        "parameters": summary["parameters"],
        "task_bias": aggregate["bias"], "task_rmse": aggregate["rmse"],
        "task_mae": aggregate["mae"], "task_corr": aggregate["corr"],
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--demo-dir", default="./reviewer_revision/results_demo/matched_input")
    parser.add_argument("--data-37-37")
    parser.add_argument("--data-137-137")
    parser.add_argument("--data-37-137")
    parser.add_argument("--out-dir", default="./reviewer_revision/results_demo/matched_input")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train", type=int, default=24)
    parser.add_argument("--max-val", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main():
    args = build_parser().parse_args()
    if args.demo:
        demo_dir = Path(args.demo_dir)
        paths = {
            "matched_37": str(demo_dir / "demo_37_to_37.h5"),
            "matched_137": str(demo_dir / "demo_137_to_137.h5"),
            "legacy_37_to_137": str(demo_dir / "demo_37_to_137.h5"),
        }
        create_demo_h5(paths["matched_37"], n_layers=37, seed=37037)
        create_demo_h5(paths["matched_137"], n_layers=137, seed=137137)
        create_demo_h5(paths["legacy_37_to_137"], n_layers=37, seed=37137)
    else:
        if not all([args.data_37_37, args.data_137_137, args.data_37_137]):
            raise SystemExit("Provide all three --data-*-* paths or use --demo.")
        paths = {
            "matched_37": args.data_37_37,
            "matched_137": args.data_137_137,
            "legacy_37_to_137": args.data_37_137,
        }
    records = [run_one(label, path, args.out_dir, args) for label, path in paths.items()]
    write_records(Path(args.out_dir) / "matched_input_summary.csv", records)
    print("Saved matched_input_summary.csv to {}".format(args.out_dir))


if __name__ == "__main__":
    main()
