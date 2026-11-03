# Data layout

The repository does not redistribute data. The scripts expect HDF5 files in the following layout.

## Revision HDF5 used by forward and Jacobian experiments

```text
train/
  clear/
    atm        [N, L, 3]       scaled vapour, temperature and pressure
    sfc        [N, 5]          surface variables
    out        [N, 22]         scaled brightness temperatures
    jac_atm    [N, 22, L, 2]   scaled dBT/d(vapour, temperature)
    jac_sfc    [N, 22, 3]      scaled dBT/d(T2m, Q2m, SP)
  cloudy/
    atm        [N, L, 3]
    sfc        [N, 5]
    hydro      [N, Lh, 4]      hydrometeor profiles
    out        [N, 22]
    jac_atm    [N, 22, L, 2]
    jac_sfc    [N, 22, 3]
    jac_cloud  [N, 22, Lh, 4]
valid/
  clear/...
  cloudy/...
```

`L` is 37 for the original experiment and may be 137 for the matched-level experiment. The Jacobian tensors can be stored in either channel-first or level-first order; `experiments/common/data_utils.py` canonicalises them.

An optional `scaler_info.pkl` may be placed next to the HDF5 file. It contains the output, atmospheric, surface and hydrometeor scaler ranges used to convert derivatives to physical units.

## Retrieval NPZ files

The LM-1DVar scripts expect one file per station, for example `50953.npz` or `prep_50953.npz`, containing:

```text
obs        [N, 22]          observed brightness temperatures
fbg        [N, 22] or Yb   background brightness temperatures
T          [N, 37]          background temperature
q          [N, 37]          background water vapour
p          [N, 37]          pressure
z          [N, 37]          height
q2m        [N]
t2m        [N]
sp or psfc [N]
clwc       [N, 37]
ciwc       [N, 37]
cswc       [N, 37]
crwc       [N, 37]
jac_T      [N, 22, 37]      optional ARMS-gb Jacobian
jac_Q      [N, 22, 37]      optional ARMS-gb Jacobian
keep       [N]              optional valid-profile mask
```

The benchmark scripts additionally use `prep_<station>.npz` with `xb`, `B`, `Yb`, `Kb`, `tclw`, `tcrw`, `clear`, `tr`, and `te`.

## Model and scaler assets

Set `QJRMS_MODEL_DIR` to a directory containing the required checkpoint files and scalers:

```text
unified_model_jacobian_finetuned.pth
bilstm.pth
mlp.pth
scaler_atm_global.pkl
scaler_sfc_global.pkl
scaler_hydro_cloudy.pkl
scaler_y.pkl
```

Original checkpoint names used by the matched-level scripts are:

```text
best_unified_model_ATT.pth
best_unified_model_BiLSTM.pth
best_unified_model_MLP.pth
best_clear_model.pth
best_cloudy_model.pth
```

## ARMS-gb

The ARMS-gb binary is external. Set `ARMS_DRV` to the compiled all-sky driver, `ARMS_FIELDS` to the field file used by the parallel evaluator, and `ARMS_ROOT` to the local ARMS-gb installation.
