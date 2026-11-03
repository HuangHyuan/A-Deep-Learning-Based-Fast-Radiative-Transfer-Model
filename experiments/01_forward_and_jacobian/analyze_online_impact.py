from pathlib import Path
import sys
_QJRMS_ROOT = Path(__file__).resolve().parents[2]
_QJRMS_COMMON = _QJRMS_ROOT / 'experiments' / 'common'
_QJRMS_CORE = _QJRMS_COMMON  # Backward-compatible alias.
if str(_QJRMS_ROOT) not in sys.path:
    sys.path.insert(0, str(_QJRMS_ROOT))
if str(_QJRMS_COMMON) not in sys.path:
    sys.path.insert(0, str(_QJRMS_COMMON))

"""Analyze paired forecast errors from control and assimilation experiments."""

import argparse
from pathlib import Path

import numpy as np

from metrics import write_records


def error_rmse(error):
    return float(np.sqrt(np.mean(np.asarray(error, dtype=np.float64) ** 2)))


def paired_improvement(truth, control, experiment, n_bootstrap=2000, seed=2026):
    control_error = np.asarray(control - truth, dtype=np.float64)
    experiment_error = np.asarray(experiment - truth, dtype=np.float64)
    baseline = error_rmse(control_error)
    updated = error_rmse(experiment_error)
    improvement = 100.0 * (baseline - updated) / baseline if baseline else float("nan")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, truth.shape[0], size=truth.shape[0])
        base = error_rmse(control_error[idx])
        exp = error_rmse(experiment_error[idx])
        draws.append(100.0 * (base - exp) / base if base else np.nan)
    return baseline, updated, improvement, float(np.nanquantile(draws, 0.025)), float(np.nanquantile(draws, 0.975))


def analyze(truth, control, experiment, labels=None, n_bootstrap=2000, seed=2026):
    truth, control, experiment = np.asarray(truth), np.asarray(control), np.asarray(experiment)
    if truth.ndim == 2:
        truth, control, experiment = truth[:, None, :], control[:, None, :], experiment[:, None, :]
        had_lead = False
    elif truth.ndim == 3:
        had_lead = True
    else:
        raise ValueError("Expected arrays with shape [cases, channels] or [cases, lead, channels].")
    rows = []
    leads = range(truth.shape[1]) if had_lead else ["all"]
    for lead in leads:
        if isinstance(lead, int):
            t, c, e = truth[:, lead], control[:, lead], experiment[:, lead]
        else:
            t, c, e = truth[:, 0], control[:, 0], experiment[:, 0]
        baseline, updated, improvement, low, high = paired_improvement(t, c, e, n_bootstrap, seed + len(rows))
        rows.append({"lead": str(lead), "n": int(t.shape[0]), "control_rmse": baseline,
                     "experiment_rmse": updated, "improvement_percent": improvement,
                     "improvement_ci_low": low, "improvement_ci_high": high})
    if labels is not None:
        for name, values in labels.items():
            for value in sorted(set(values.tolist())):
                mask = values == value
                baseline, updated, improvement, low, high = paired_improvement(
                    truth[mask], control[mask], experiment[mask], n_bootstrap, seed + len(rows)
                )
                rows.append({"lead": "all", "group_type": name, "group": str(value),
                             "n": int(mask.sum()), "control_rmse": baseline,
                             "experiment_rmse": updated, "improvement_percent": improvement,
                             "improvement_ci_low": low, "improvement_ci_high": high})
    return rows


def create_demo(n=120, leads=5, channels=4, seed=2026):
    rng = np.random.default_rng(seed)
    truth = rng.normal(0.0, 1.0, size=(n, leads, channels)).astype(np.float32)
    control = truth + rng.normal(0.0, 0.35, size=truth.shape).astype(np.float32)
    experiment = truth + rng.normal(0.0, 0.31, size=truth.shape).astype(np.float32)
    month = np.array(["Jan", "Feb", "Mar"])[rng.integers(0, 3, size=n)]
    return {"truth": truth, "control": control, "experiment": experiment, "month": month}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="NPZ with truth, control and experiment arrays.")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--out-dir", default="./reviewer_revision/results_demo/online_impact")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main():
    args = build_parser().parse_args()
    if args.input:
        data = np.load(args.input)
        truth, control, experiment = data["truth"], data["control"], data["experiment"]
        labels = {key: data[key] for key in ("month", "region") if key in data}
    elif args.demo:
        data = create_demo(seed=args.seed)
        truth, control, experiment = data["truth"], data["control"], data["experiment"]
        labels = {"month": data["month"]}
    else:
        raise SystemExit("Provide --input or use --demo.")
    rows = analyze(truth, control, experiment, labels, args.bootstrap, args.seed)
    write_records(Path(args.out_dir) / "online_impact_metrics.csv", rows)
    print("Saved online_impact_metrics.csv to {}".format(args.out_dir))


if __name__ == "__main__":
    main()
