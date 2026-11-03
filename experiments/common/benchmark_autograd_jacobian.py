from pathlib import Path
import sys
_QJRMS_ROOT = Path(__file__).resolve().parents[2]
_QJRMS_COMMON = _QJRMS_ROOT / 'experiments' / 'common'
_QJRMS_CORE = _QJRMS_COMMON  # Backward-compatible alias.
if str(_QJRMS_ROOT) not in sys.path:
    sys.path.insert(0, str(_QJRMS_ROOT))
if str(_QJRMS_COMMON) not in sys.path:
    sys.path.insert(0, str(_QJRMS_COMMON))

"""Benchmark the autograd-loop Jacobian implementation."""

import argparse
import sys
import time

import torch


from GB_model import UnifiedAllSkyRTM_BiLSTM, UnifiedAllSkyRTM_MLP


def compute_jacobian_autograd(model, inputs_dict, is_create_graph=False):
    processed_inputs = {}
    for key, tensor in inputs_dict.items():
        if tensor is not None:
            t = tensor.clone().requires_grad_(True)
            t.retain_grad()
            processed_inputs[key] = t
    lstm_modules = [
        m for m in model.modules()
        if isinstance(m, (torch.nn.LSTM, torch.nn.RNN, torch.nn.GRU,
                          torch.nn.LSTMCell, torch.nn.RNNCell, torch.nn.GRUCell))
    ]
    original_modes = [m.training for m in lstm_modules]
    for module in lstm_modules:
        module.train()
    try:
        if inputs_dict.get("hydro") is not None:
            outputs = model(processed_inputs["atm"], processed_inputs["sfc"], processed_inputs["hydro"])
        else:
            outputs = model(processed_inputs["atm"], processed_inputs["sfc"])
        batch, channels = outputs.shape
        jacobians = {}
        for key, tensor in processed_inputs.items():
            shape = tensor.shape[1:]
            jacobians[key] = torch.zeros((batch, channels) + tuple(shape), device=tensor.device)
        for channel in range(channels):
            model.zero_grad()
            for tensor in processed_inputs.values():
                if tensor.grad is not None:
                    tensor.grad.zero_()
            grad_outputs = torch.zeros_like(outputs)
            grad_outputs[:, channel] = 1.0
            grads = torch.autograd.grad(
                outputs, list(processed_inputs.values()), grad_outputs=grad_outputs,
                retain_graph=True if is_create_graph else channel < channels - 1,
                create_graph=is_create_graph, allow_unused=False,
            )
            for index, key in enumerate(processed_inputs):
                jacobians[key][:, channel, ...] = grads[index]
    finally:
        for module, mode in zip(lstm_modules, original_modes):
            module.train(mode)
    return jacobians


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["bilstm", "mlp"], required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--create-graph", action="store_true")
    parser.add_argument("--cloudy", action="store_true")
    parser.add_argument("--iters", type=int, default=1)
    args = parser.parse_args()

    if args.create_graph:
        torch.backends.cudnn.enabled = False
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except AttributeError:
            pass
    if args.model == "bilstm":
        model = UnifiedAllSkyRTM_BiLSTM(hidden_dim=180, num_bi_lstm_blocks=2)
    else:
        model = UnifiedAllSkyRTM_MLP(hidden_dim=12, num_mlp_blocks=3)
    model = model.cuda().eval()
    atmos = torch.randn(args.batch_size, 37, 3, device="cuda")
    surf = torch.randn(args.batch_size, 5, device="cuda")
    hydro = torch.randn(args.batch_size, 36, 4, device="cuda") if args.cloudy else None
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(args.iters):
        jac = compute_jacobian_autograd(
            model, {"atm": atmos, "sfc": surf, "hydro": hydro},
            is_create_graph=args.create_graph,
        )
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print("model", args.model, "batch", args.batch_size, "cloudy", args.cloudy,
          "create_graph", args.create_graph, "seconds_per_batch", elapsed / args.iters,
          "seconds_per_sample", elapsed / (args.iters * args.batch_size),
          "atm_jac", tuple(jac["atm"].shape))


if __name__ == "__main__":
    main()
