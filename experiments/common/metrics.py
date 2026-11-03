"""Metrics and analytical Jacobian evaluation for revision experiments."""

from pathlib import Path
import csv
import numpy as np


def write_records(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in records for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def _safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _safe_cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float("nan") if denom < 1e-12 else float(np.dot(a, b) / denom)


def numerical_metrics(truth, prediction):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - truth
    return {
        "n": int(truth.size),
        "bias": float(np.mean(error)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "corr": _safe_corr(truth, prediction),
        "cosine": _safe_cosine(truth, prediction),
        "truth_rms": float(np.sqrt(np.mean(truth ** 2))),
    }


def task_metrics(model_name, split, truth, prediction):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    rows = [{"model": model_name, "split": split, "channel": "all",
             **numerical_metrics(truth, prediction)}]
    for channel in range(truth.shape[1]):
        rows.append({"model": model_name, "split": split, "channel": int(channel),
                     **numerical_metrics(truth[:, channel], prediction[:, channel])})
    return rows


def jacobian_metrics(model_name, split, variable, truth, prediction, include_levels=False):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape:
        raise ValueError("Jacobian shape mismatch: {} vs {}".format(truth.shape, prediction.shape))
    rows = []
    if truth.ndim == 4:
        _, n_levels, n_channels, n_variables = truth.shape
        for channel in range(n_channels):
            for variable_idx in range(n_variables):
                rows.append({
                    "model": model_name, "split": split, "jacobian": variable,
                    "channel": int(channel), "component": int(variable_idx), "level": "all",
                    **numerical_metrics(truth[:, :, channel, variable_idx],
                                        prediction[:, :, channel, variable_idx]),
                })
        if include_levels:
            for channel in range(n_channels):
                for variable_idx in range(n_variables):
                    for level_idx in range(n_levels):
                        rows.append({
                            "model": model_name, "split": split, "jacobian": variable,
                            "channel": int(channel), "component": int(variable_idx),
                            "level": int(level_idx),
                            **numerical_metrics(truth[:, level_idx, channel, variable_idx],
                                                prediction[:, level_idx, channel, variable_idx]),
                        })
    elif truth.ndim == 3:
        _, n_channels, n_variables = truth.shape
        for channel in range(n_channels):
            for variable_idx in range(n_variables):
                rows.append({
                    "model": model_name, "split": split, "jacobian": variable,
                    "channel": int(channel), "component": int(variable_idx), "level": "surface",
                    **numerical_metrics(truth[:, channel, variable_idx],
                                        prediction[:, channel, variable_idx]),
                })
    else:
        raise ValueError("Unsupported Jacobian rank: {}".format(truth.ndim))
    return rows
