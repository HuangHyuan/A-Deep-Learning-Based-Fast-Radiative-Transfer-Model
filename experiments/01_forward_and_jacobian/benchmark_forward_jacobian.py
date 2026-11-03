"""Benchmark forward and Jacobian evaluation using the manuscript protocol.

The timing excludes host-to-device transfer: input tensors are created on the
target device before timing starts. CPU runs default to eight threads, forward
benchmarks use a batch of 5,120 profiles, and Jacobian benchmarks use 64.
"""
from pathlib import Path
import sys

_QJRMS_ROOT = Path(__file__).resolve().parents[2]
_QJRMS_COMMON = _QJRMS_ROOT / "experiments" / "common"
if str(_QJRMS_ROOT) not in sys.path:
    sys.path.insert(0, str(_QJRMS_ROOT))
if str(_QJRMS_COMMON) not in sys.path:
    sys.path.insert(0, str(_QJRMS_COMMON))

import argparse
import csv
import json
import platform
import time

import torch

from manuscript_config import BENCHMARK_PROTOCOL, MODEL_CONFIGS
from revised_model import build_model, count_parameters
from training_utils import resolve_device


def model_kwargs(name, n_layers=37, n_channels=22):
    config = MODEL_CONFIGS[name]
    kwargs = {
        "n_layers": n_layers,
        "n_channels": n_channels,
        "hidden_dim": config["hidden_dim"],
    }
    if name == "transformer":
        kwargs.update({
            "nhead": config["nhead"],
            "num_transformer_layers": config["num_transformer_layers"],
        })
    else:
        kwargs["num_layers"] = config["num_layers"]
    return kwargs


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def forward_once(model, atm, sfc, hydro):
    with torch.inference_mode():
        if hydro is None:
            return model(atm, sfc)
        return model(atm, sfc, hydro)


def jacobian_once(model, name, atm, sfc, hydro):
    if name in {"transformer", "mlp"}:
        if hydro is None:
            fn = lambda x, s: model(x.unsqueeze(0), s.unsqueeze(0)).squeeze(0)
            return torch.func.vmap(torch.func.jacrev(fn, argnums=0))(atm, sfc)
        fn = lambda x, s, h: model(
            x.unsqueeze(0), s.unsqueeze(0), h.unsqueeze(0)
        ).squeeze(0)
        return torch.func.vmap(torch.func.jacrev(fn, argnums=0))(atm, sfc, hydro)

    # cuDNN recurrent backward is only available in training mode. BiLSTM and
    # the simple LSTM use zero dropout in the released configuration, so their
    # forward values remain deterministic.
    was_training = model.training
    model.train()
    try:
        atm = atm.detach().clone().requires_grad_(True)
        if hydro is None:
            output = model(atm, sfc)
        else:
            output = model(atm, sfc, hydro)
        jacobian = torch.empty(
            (atm.shape[0], output.shape[1], atm.shape[1], atm.shape[2]),
            device=atm.device, dtype=atm.dtype,
        )
        for channel in range(output.shape[1]):
            jacobian[:, channel] = torch.autograd.grad(
                output[:, channel].sum(), atm, retain_graph=channel < output.shape[1] - 1
            )[0]
        return jacobian
    finally:
        model.train(was_training)


def timed_call(call, device, warmup, repeats):
    for _ in range(warmup):
        call()
    sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        call()
    sync(device)
    return (time.perf_counter() - start) / max(repeats, 1)


def benchmark(name, device, scenes, threads, forward_batch, jacobian_batch,
              warmup, repeats):
    model = build_model(name, **model_kwargs(name)).to(device).eval()
    rows = []
    for scene in scenes:
        hydro = torch.rand(forward_batch, 36, 4, device=device) if scene == "cloudy" else None
        atm = torch.randn(forward_batch, 37, 3, device=device)
        sfc = torch.randn(forward_batch, 5, device=device)
        forward_seconds = timed_call(
            lambda: forward_once(model, atm, sfc, hydro), device, warmup, repeats
        )
        rows.append({
            "model": name, "hardware": str(device), "threads": threads,
            "scene": scene, "kind": "forward", "batch_size": forward_batch,
            "seconds": forward_seconds, "parameters": count_parameters(model),
        })

        hydro = torch.rand(jacobian_batch, 36, 4, device=device) if scene == "cloudy" else None
        atm = torch.randn(jacobian_batch, 37, 3, device=device)
        sfc = torch.randn(jacobian_batch, 5, device=device)
        jacobian_seconds = timed_call(
            lambda: jacobian_once(model, name, atm, sfc, hydro), device, warmup, repeats
        )
        rows.append({
            "model": name, "hardware": str(device), "threads": threads,
            "scene": scene, "kind": "jacobian", "batch_size": jacobian_batch,
            "seconds": jacobian_seconds, "parameters": count_parameters(model),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="transformer,bilstm,mlp")
    parser.add_argument("--scenes", default="clear,cloudy")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=BENCHMARK_PROTOCOL["cpu_threads"])
    parser.add_argument("--forward-batch", type=int, default=BENCHMARK_PROTOCOL["forward_batch_size"])
    parser.add_argument("--jacobian-batch", type=int, default=BENCHMARK_PROTOCOL["jacobian_batch_size"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-dir", default="./outputs/benchmark")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    rows = []
    for name in [item.strip() for item in args.models.split(",") if item.strip()]:
        rows.extend(benchmark(
            name, device, [item.strip() for item in args.scenes.split(",") if item.strip()],
            args.threads, args.forward_batch, args.jacobian_batch,
            args.warmup, args.repeats,
        ))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "forward_jacobian_timing.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "device": str(device),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cpu_threads": args.threads,
        "host_to_device_transfer_timed": False,
        "forward_batch_size": args.forward_batch,
        "jacobian_batch_size": args.jacobian_batch,
        "model_configs": MODEL_CONFIGS,
    }
    (out / "benchmark_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved benchmark results to", out)


if __name__ == "__main__":
    main()
