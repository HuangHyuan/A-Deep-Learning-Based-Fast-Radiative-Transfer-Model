# ARMS-gb reference model

- `armsgb_driver_allsy.f90`: all-sky Fortran driver for forward brightness temperature and Jacobian output.
- `armsgb_allsy.py`: Python text-file adapter for one ARMS-gb calculation.
- `compute_arms_all_profiles_agg_valid.py`: aggregates ARMS-gb reference Jacobian statistics.
- The parallel evaluator is shared as `experiments/common/arms_batch.py`.

The ARMS-gb base library is not redistributed. Compile the driver against the local ARMS-gb libraries and set `ARMS_DRV` to the resulting executable.

The driver loads ARMS-gb sensor coefficients from `./fix/`. Place the licensed coefficient files there before compiling or running.

Codes of ARMS-gb v2.0 are available at https://doi.org/10.5281/zenodo.17318669 (Huang Z. et al., 2025).
