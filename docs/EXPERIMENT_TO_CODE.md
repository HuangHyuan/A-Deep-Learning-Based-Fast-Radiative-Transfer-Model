# Experiment-to-code map

| Manuscript/response item | Experiment | Main entry point | Supporting code |
|---|---|---|---|
| Forward-model comparison | Transformer, BiLSTM, MLP and Simple-LSTM trained on matched inputs | `experiments/01_forward_and_jacobian/train_models.py` | `GB_model.py`, `experiments/common/revised_model.py` |
| Forward evaluation | Channel-wise RMSE, bias and correlation | `experiments/01_forward_and_jacobian/evaluate_models.py` | `experiments/common/metrics.py`, `experiments/common/data_utils.py` |
| Jacobian quantitative evaluation | Analytical Jacobian RMSE and correlation by scene/channel/variable | `experiments/01_forward_and_jacobian/evaluate_models.py` and `evaluate_jacobian_three.py` | `experiments/common/training_utils.py` |
| Jacobian computation | Five-station Jacobian statistics and profile aggregation | `experiments/03_jacobian_evaluation/compute_five_station_jacobian_full.py` | `run_all_profiles_jacobian_local.py` |
| 37-to-37 and 137-to-137 matching | Level-matched brightness-temperature experiments | `experiments/02_level_matching/run_matched_input.py` | `train_original_bt.py`, `evaluate_original_matched.py` |
| 1D-Var/OEM retrieval | Five-station all-sky retrieval | `experiments/04_data_assimilation/run_station_ret.py` | `run_station_prep.py` |
| CPU/GPU timing | Forward and Jacobian benchmark | `experiments/01_forward_and_jacobian/benchmark_forward_jacobian.py` | `experiments/common/manuscript_config.py` |
| CPU/GPU retrieval timing | Full three-iteration LM-1DVar timing | `experiments/04_data_assimilation/benchmark_retrieval_autograd.py` | `cpu_retrieval_benchmark.py` (archived prototype) |
| Scene-context/gate ablation | Mean vs sum context, learned/fixed gate, single/shared head | `experiments/05_scene_gate_ablation/train_scene_gate_ablation.py` | `GB_model.py` |
| ARMS-gb reference generation | All-sky forward/Jacobian calculations | `experiments/06_armsgb_reference/armsgb_allsy.py` | `experiments/common/arms_batch.py`, `armsgb_driver_allsy.f90` |
| Data preparation | HDF5 construction and Jacobian subset generation | `experiments/07_data_preparation/` | `configs/prepare_config.example.json` |

## Jacobian implementation

- Transformer and MLP: `torch.func.vmap(torch.func.jacrev(...))`.
- BiLSTM: batched reverse-mode `torch.autograd.grad` for each output channel.
- The BiLSTM temporary training-mode switch is required by cuDNN for recurrent backward propagation. All released BiLSTM dropout probabilities are zero, so forward values remain deterministic.
