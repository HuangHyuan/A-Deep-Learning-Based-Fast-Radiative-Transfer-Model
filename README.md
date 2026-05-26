# A-Deep-Learning-Based-Fast-Radiative-Transfer-Model
**Deep learning based fast radiative transfer model for all‑sky satellite observations.**
This repository provides a PyTorch implementation of a **Transformer‑based unified all‑sky radiative transfer model** that simulates satellite brightness temperatures(BT) from atmospheric profiles, surface conditions, and hydrometeors (cloud liquid/ice, rain, snow). It supports both clear‑sky and cloudy‑sky scenes in a single forward pass, and optionally enforces physical consistency via **Jacobian‑constrained fine‑tuning**.

# Repository Structure
```bash
.
├── GB_model.py # Transformer model definition
├── GB_FW_Train.py # Standard training (BT only)
├── GB_J_Train.py # Jacobian‑constrained fine‑tuning
├── Generate_FW_Dataset.py # Build HDF5 dataset (inputs + BT)
├── Generate_J_Dataset.py # Build HDF5 dataset (inputs + BT + Jacobians)
└── scalers/ # Saved MinMaxScalers and Jacobian stats

# Requirements
Python 3.8+
PyTorch >= 1.12 (with torch.func support)
h5py, netCDF4, numpy, scikit‑learn, tqdm, joblib

