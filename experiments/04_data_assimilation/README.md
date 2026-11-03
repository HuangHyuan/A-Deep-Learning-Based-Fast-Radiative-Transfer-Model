# LM-1DVar data assimilation

Final manuscript workflow:

- `run_station_ret.py`: station-level LM-1DVar retrieval using ARMS-gb and the Transformer operator, with three iterations by default.
- `run_station_prep.py`: prepares station data and ARMS-gb reference fields.
- `benchmark_retrieval_autograd.py`: CPU/GPU timing for the final three-iteration retrieval. Transformer and MLP use `vmap+jacrev`; BiLSTM uses batched reverse-mode `autograd.grad`.
- `prep_sounding.py`, `validate_sounding.py`: radiosonde preparation and validation.

Archived prototypes, retained only for provenance:

- `retrieval_server.py`: compact one-step, single-pass retrieval example.
- `cpu_retrieval_benchmark.py`: earlier CPU-only benchmark using finite-difference BiLSTM Jacobians.

Set `QJRMS_RETRIEVAL_DATA`, `QJRMS_RETRIEVAL_WORK`, `QJRMS_MODEL_DIR` and the ARMS environment variables described in `docs/DATA_LAYOUT.md`.
