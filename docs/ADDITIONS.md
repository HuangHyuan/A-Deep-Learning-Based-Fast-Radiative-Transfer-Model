# Additions relative to the original repository

## Models and configuration

- Extended `GB_model.py` with BiLSTM, MLP and compact vertical LSTM implementations.
- Added a unified model factory and the model configurations reported in the manuscript.
- Confirmed support for both 37-level and 137-level vertical grids.

## Experiment code

- Added multi-architecture training, forward evaluation and quantitative Jacobian evaluation.
- Added 37-to-37, 137-to-137 and 37-to-137 matched-level experiments.
- Added five-station Jacobian computation and aggregation.
- Added LM-1DVar retrieval and CPU/GPU timing code.
- Added scene-context and gate ablation experiments.
- Added ARMS-gb reference-generation wrappers and the all-sky Fortran driver.
- Added HDF5 construction and Jacobian-subset preparation tools.

## Reproducibility

- Added `requirements.txt`, path templates, data-layout documentation and a file inventory.
- Added a data-free smoke test covering all architectures and both vertical grids.
- Added a GitHub Actions workflow for the smoke test.
- Added the ARMS-gb v2.0 code-availability statement and Zenodo DOI.
