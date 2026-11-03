# A-Deep-Learning-Based-Fast-Radiative-Transfer-Model

PyTorch implementations of fast radiative-transfer emulators for all-sky satellite brightness-temperature simulation, Jacobian evaluation, and 1D-Var retrieval experiments.

The repository contains the experiment code used in the manuscript and revision.

The repository includes the original Transformer implementation and all evaluated architectures:

- Transformer: `GB_model.UnifiedAllSkyRTM_Transformer`
- bidirectional LSTM: `GB_model.UnifiedAllSkyRTM_BiLSTM`
- MLP: `GB_model.UnifiedAllSkyRTM_MLP`
- simple vertical LSTM: `GB_model.SimpleVerticalLSTM`

`GB_model.build_model()` provides a single model factory for `transformer`, `bilstm`, `mlp`, and `simple_lstm`.

| Model | Manuscript configuration | Parameters |
|---|---|---:|
| Transformer | 3 blocks, hidden size 160, 4 heads | 943,384 |
| BiLSTM | 2 blocks, hidden size 180 | 932,474 |
| MLP | 3 blocks, hidden size 12 | 1,189,406 |
| Simple-LSTM | 2 recurrent layers, hidden size 128 | 370,070 |

## Additions relative to the original repository

- Added BiLSTM, MLP and compact vertical LSTM models with the configurations reported in the manuscript.
- Added multi-architecture training, evaluation, Jacobian and benchmark code.
- Added matched 37/137-level, data-assimilation, gate-ablation, ARMS-gb and data-preparation experiments.
- Added reproducibility documentation, path templates, requirements and a data-free smoke test.

See `docs/ADDITIONS.md` for the complete list.

## Repository layout

```text
GB_model.py                         Model definitions and unified model factory
GB_FW_Train.py                      Original Transformer training loop
GB_J_Train.py                       Transformer Jacobian-constrained fine-tuning
Generate_FW_Dataset.py              Brightness-temperature dataset builder
Generate_J_Dataset.py               Brightness-temperature and Jacobian dataset builder

experiments/
  common/                           Shared data, metric, training and model utilities
  01_forward_and_jacobian/          Multi-architecture training and Jacobian evaluation
  02_level_matching/                37-to-37, 137-to-137 and 37-to-137 comparisons
  03_jacobian_evaluation/           Jacobian computation and evaluation
  04_data_assimilation/             LM-1DVar retrieval and CPU/GPU timing
  05_scene_gate_ablation/           Cloud-context and gate ablations
  06_armsgb_reference/              ARMS-gb Python wrappers and Fortran driver
  07_data_preparation/              HDF5 construction and Jacobian subset tools

configs/                            Example data/model path configuration
docs/                               Data layout and experiment-to-code map
tests/smoke_test_models.py          Data-free architecture verification
```

## Requirements

```bash
pip install -r requirements.txt
```

The released experiments were run with Python 3.12, PyTorch 2.8, CUDA 12.8, h5py, NumPy, scikit-learn, joblib, pandas and netCDF4.

## Quick verification without data

```bash
python tests/smoke_test_models.py
```

Expected final line:

```text
ALL_SMOKE_TESTS_PASSED
```

The test checks forward and cloudy-sky predictions for the Transformer, BiLSTM, MLP and simple LSTM, exercises the model factory, and verifies reverse-mode `autograd.grad` through the BiLSTM.

## Training multiple architectures

The original training script `GB_FW_Train.py` remains available for the Transformer. The revision training entry point supports all architectures:

```bash
python experiments/01_forward_and_jacobian/train_models.py \
  --data /path/to/revision_data.h5 \
  --models transformer,bilstm,mlp,simple_lstm \
  --out-dir /path/to/runs \
  --epochs 100 --batch-size 256 --device cuda
```

A data-free smoke experiment is also available:

```bash
python experiments/01_forward_and_jacobian/train_models.py --demo \
  --models transformer,bilstm,mlp,simple_lstm \
  --epochs 1 --batch-size 4 --max-train 8 --max-val 4 \
  --hidden-dim 16 --num-layers 1 --device cpu
```

## Jacobian evaluation

```bash
python experiments/01_forward_and_jacobian/evaluate_models.py --help
python experiments/01_forward_and_jacobian/evaluate_jacobian_three.py --help
```

The implementation follows the manuscript setup:

- Transformer and MLP Jacobians use `vmap + jacrev`.
- BiLSTM Jacobians use batched reverse-mode `autograd.grad`.
- The BiLSTM temporary training-mode switch is required by cuDNN for recurrent backward propagation. All released BiLSTM dropout probabilities are zero, so forward values remain deterministic.

## Data assimilation and timing

```bash
python experiments/04_data_assimilation/run_station_ret.py --help
python experiments/04_data_assimilation/benchmark_retrieval_autograd.py --device cpu
python experiments/04_data_assimilation/benchmark_retrieval_autograd.py --device gpu
```

The timing script covers the complete three-iteration LM-1DVar loop, including forward calls, Jacobian evaluation, linear algebra and line-search forward calls. Model loading and file I/O are excluded. `retrieval_server.py` and `cpu_retrieval_benchmark.py` are archived prototypes and are not used for the final manuscript timing values.

## Scene-context and gate ablation

```bash
python experiments/05_scene_gate_ablation/train_scene_gate_ablation.py --help
```

## ARMS-gb reference model

`experiments/06_armsgb_reference/` contains the Python adapters, parallel evaluator and all-sky Fortran driver. The ARMS-gb base library, sensor coefficients and compiled binary are not redistributed.

Code availability: Codes of ARMS-gb v2.0 are available at https://doi.org/10.5281/zenodo.17318669 (Huang Z. et al., 2025).

Set `ARMS_DRV`, `ARMS_FIELDS` and `ARMS_ROOT` for a local installation.

## Data and checkpoints

No training data, processed HDF5/NetCDF files, model weights or scalers are included. Data and checkpoints should be stored outside the repository and referenced with the environment variables in `configs/paths.example.sh` or `configs/paths.example.ps1`.

See `docs/DATA_LAYOUT.md` for dataset schemas, `docs/EXPERIMENT_TO_CODE.md` for the experiment-to-code map, and `docs/ADDITIONS.md` for additions relative to the original repository.

The five-station experiments use stations `[50953, 52203, 59097, 58238, 54623]`.
