# Scene-context and gate ablation

`train_scene_gate_ablation.py` trains four all-sky L37-input/L137-target Transformer variants:

- baseline: mean hydrometeor context + learned gate;
- sum-context: sum hydrometeor context + learned gate;
- fixed-gate: fixed 0.5/0.5 clear/cloudy weighting;
- single-head: one shared output head without scene gating.

Set `QJRMS_DATA_L37`, `QJRMS_DATA_L137` and `QJRMS_RUNS_DIR`.
