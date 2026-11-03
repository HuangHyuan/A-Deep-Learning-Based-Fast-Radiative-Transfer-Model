# Forward-model and Jacobian experiments

Main scripts:

- `train_models.py`: trains Transformer, BiLSTM, MLP and Simple-LSTM under one protocol.
- `evaluate_models.py`: computes forward brightness-temperature metrics and analytical Jacobian metrics.
- `benchmark_models.py`: parameter count, latency, throughput and memory benchmark.
- `evaluate_jacobian_three.py`: notebook-style Jacobian evaluation for Transformer, BiLSTM and MLP.
- `evaluate_generalization.py`: location/season/weather-regime metrics with bootstrap intervals.
- `analyze_online_impact.py`: paired control-versus-experiment analysis when online pairs exist.

Quick demo:

```bash
python experiments/01_forward_and_jacobian/train_models.py --demo \
  --models transformer,bilstm,mlp,simple_lstm --epochs 1 --batch-size 4 \
  --max-train 8 --max-val 4 --hidden-dim 16 --num-layers 1 --device cpu
```
