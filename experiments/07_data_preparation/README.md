# Data preparation

- `prepare_revision_h5.py`: builds the unified revision HDF5 from text/NetCDF/Jacobian inputs.
- `prepare_revision_bt_h5.py`: builds matched 37/137-level brightness-temperature datasets.
- `convert_existing_h5.py`: converts an existing HDF5 and scaler file to the revision layout.
- `build_jacobian_subset.py`: creates a compact Jacobian training subset.

Use `configs/prepare_config.example.json` as the starting point. The repository does not include source data.
