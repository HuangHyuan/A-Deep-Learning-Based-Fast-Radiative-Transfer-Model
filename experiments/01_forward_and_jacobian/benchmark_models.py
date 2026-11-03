from pathlib import Path
import sys
_QJRMS_ROOT = Path(__file__).resolve().parents[2]
_QJRMS_COMMON = _QJRMS_ROOT / 'experiments' / 'common'
_QJRMS_CORE = _QJRMS_COMMON  # Backward-compatible alias.
if str(_QJRMS_ROOT) not in sys.path:
    sys.path.insert(0, str(_QJRMS_ROOT))
if str(_QJRMS_COMMON) not in sys.path:
    sys.path.insert(0, str(_QJRMS_COMMON))

"""Benchmark inference latency/throughput and hardware metadata."""

import argparse
import json
import platform
import time
from pathlib import Path

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch

from manuscript_config import BENCHMARK_PROTOCOL, MODEL_CONFIGS
from revised_model import build_model, count_parameters
from training_utils import resolve_device, set_seed


def hardware_metadata(device):
    metadata = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        metadata["gpu_name"] = torch.cuda.get_device_name(device)
        metadata["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(device).total_memory
    return metadata


def model_config(name, layers, channels, hidden_dim=None, num_layers=None, nhead=None):
    config = MODEL_CONFIGS[name]
    kwargs = {
        "n_layers": layers,
        "n_channels": channels,
        "hidden_dim": hidden_dim or config["hidden_dim"],
    }
    if name == "transformer":
        kwargs.update({
            "nhead": nhead or config["nhead"],
            "num_transformer_layers": num_layers or config["num_transformer_layers"],
        })
    elif name in {"bilstm", "mlp", "simple_lstm"}:
        kwargs["num_layers"] = num_layers or config["num_layers"]
    return kwargs


def benchmark_model(name, layers, channels, hidden_dim, num_layers, nhead,
                    batch_sizes, device, warmup, repeats, precision, seed):
    set_seed(seed)
    kwargs = model_config(name, layers, channels, hidden_dim, num_layers, nhead)
    model = build_model(name, **kwargs).to(device).eval()
    rows = []
    for batch_size in batch_sizes:
        atm = torch.randn(batch_size, layers, 3, device=device)
        sfc = torch.randn(batch_size, 5, device=device)
        hydro = torch.rand(batch_size, layers, 4, device=device)
        with torch.inference_mode():
            for _ in range(warmup):
                model(atm, sfc, hydro)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(repeats):
                model(atm, sfc, hydro)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory = torch.cuda.max_memory_allocated(device)
        else:
            peak_memory = float("nan")
        elapsed = time.perf_counter() - start
        milliseconds = elapsed * 1000.0 / repeats
        rows.append({
            "model": name, "device": str(device), "precision": precision,
            "n_layers": layers, "n_channels": channels,
            "batch_size": batch_size, "parameters": count_parameters(model),
            "latency_ms_per_batch": milliseconds,
            "throughput_samples_per_second": batch_size * repeats / elapsed,
            "peak_memory_bytes": peak_memory,
        })
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="transformer,bilstm,mlp,simple_lstm")
    parser.add_argument("--n-layers", type=int, default=37)
    parser.add_argument("--n-channels", type=int, default=22)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--batch-sizes", default=str(BENCHMARK_PROTOCOL["forward_batch_size"]))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["fp32", "amp-fp16"], default="fp32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--threads", type=int, default=BENCHMARK_PROTOCOL["cpu_threads"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-dir", default="./outputs/benchmark_forward")
    return parser


def main():
    args = build_parser().parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item.strip()]
    rows = []
    for name in [item.strip() for item in args.models.split(",") if item.strip()]:
        rows.extend(benchmark_model(
            name, args.n_layers, args.n_channels, args.hidden_dim, args.num_layers,
            args.nhead, batch_sizes, device, args.warmup, args.repeats,
            args.precision, args.seed,
        ))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with (out / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = __import__("csv").DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (out / "hardware.json").write_text(
        json.dumps(hardware_metadata(device), indent=2), encoding="utf-8"
    )
    print("Saved benchmark.csv and hardware.json to {}".format(out))


if __name__ == "__main__":
    main()
