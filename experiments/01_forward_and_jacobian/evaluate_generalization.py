from pathlib import Path
import sys
_QJRMS_ROOT = Path(__file__).resolve().parents[2]
_QJRMS_COMMON = _QJRMS_ROOT / 'experiments' / 'common'
_QJRMS_CORE = _QJRMS_COMMON  # Backward-compatible alias.
if str(_QJRMS_ROOT) not in sys.path:
    sys.path.insert(0, str(_QJRMS_ROOT))
if str(_QJRMS_COMMON) not in sys.path:
    sys.path.insert(0, str(_QJRMS_COMMON))

"""Grouped generalization metrics with bootstrap confidence intervals."""

import argparse
from pathlib import Path

import numpy as np

from metrics import numerical_metrics, write_records


def bootstrap_interval(truth, prediction, n_bootstrap=1000, seed=2026, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = truth.shape[0]
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        error = prediction[idx] - truth[idx]
        values.append(float(np.sqrt(np.mean(error ** 2))))
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def summarize_group(label_name, label_value, truth, prediction, n_bootstrap, seed):
    metrics = numerical_metrics(truth, prediction)
    low, high = bootstrap_interval(truth, prediction, n_bootstrap, seed)
    return {"group_type": label_name, "group": str(label_value), **metrics,
            "rmse_ci_low": low, "rmse_ci_high": high}


def evaluate_groups(truth, prediction, metadata, n_bootstrap=1000, seed=2026):
    rows = [summarize_group("overall", "all", truth, prediction, n_bootstrap, seed)]
    for key, values in metadata.items():
        for value in sorted(set(values.tolist())):
            mask = values == value
            if mask.sum() >= 2:
                rows.append(summarize_group(key, value, truth[mask], prediction[mask],
                                            n_bootstrap, seed + len(rows)))
    return rows


def create_demo_predictions(seed=2026, n=180, channels=22):
    rng = np.random.default_rng(seed)
    truth = rng.normal(240.0, 8.0, size=(n, channels)).astype(np.float32)
    prediction = truth + rng.normal(0.0, 1.0, size=truth.shape).astype(np.float32)
    locations = np.array(["coast", "inland", "plateau"])[rng.integers(0, 3, size=n)]
    seasons = np.array(["DJF", "MAM", "JJA", "SON"])[rng.integers(0, 4, size=n)]
    regimes = np.array(["clear", "cloudy", "precipitating"])[rng.integers(0, 3, size=n)]
    return {"truth": truth, "prediction": prediction, "location": locations,
            "season": seasons, "regime": regimes}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="NPZ with truth and prediction arrays plus group labels.")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--out-dir", default="./reviewer_revision/results_demo/generalization")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main():
    args = build_parser().parse_args()
    if args.input:
        data = np.load(args.input)
        truth, prediction = data["truth"], data["prediction"]
        metadata = {key: data[key] for key in ("location", "season", "regime") if key in data}
    elif args.demo:
        data = create_demo_predictions(args.seed)
        truth, prediction = data["truth"], data["prediction"]
        metadata = {key: data[key] for key in ("location", "season", "regime")}
    else:
        raise SystemExit("Provide --input or use --demo.")
    rows = evaluate_groups(truth, prediction, metadata, args.bootstrap, args.seed)
    write_records(Path(args.out_dir) / "generalization_summary.csv", rows)
    print("Saved generalization_summary.csv to {}".format(args.out_dir))


if __name__ == "__main__":
    main()
