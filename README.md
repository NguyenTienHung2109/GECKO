# GECKO

Federated continual graph learning benchmark: source in `src/gecko/`, configurations
and **228 predefined split cells** in `reproduce/`. Model weights are not included.

## Install

```bash
conda env create -f reproduce/environment.yml
conda activate gecko
python -m pip install --no-deps -e .
python -m pip install --only-binary=:all: 'torch-scatter==2.1.2+pt27cpu' \
  -f https://data.pyg.org/whl/torch-2.7.0+cpu.html
python -m pip install -e '.[graph,pyg,legacy-gem]'
```

These commands use CPU. CUDA requires matching PyTorch, DGL and extension wheels.

## Use the supplied splits

```bash
python reproduce/check.py
python -m gecko.data.predefined list
python -m gecko.data.predefined materialize --id S5-s0-a100-synchronized \
  --data-root data --output generated_streams/predefined
python reproduce/run.py --id S5-s0-a100-synchronized --method GEM --device cpu
```

`reproduce/splits/` contains saved client assignments, train/validation/test queries,
task orders and participation schedules. Import restores them without resampling
and verifies their hashes. Compact LC/LP data are included; NC raw data are
loaded/downloaded to `--data-root`. The catalog lists available scenarios, seeds
and settings. Training outputs go to `generated_streams/predefined/`.

## Construct your own splits

```bash
python reproduce/construct.py --scenarios S2 --data-root data --output outputs/new
python -m gecko.benchmarks.paper --stage run --scenarios S2 --methods GEM \
  --device cpu --output outputs/new
```

Use `--help` for options, or `python reproduce/smoke.py` for a small synthetic CPU run.
New construction follows the recipe; use saved splits for the exact released assignments.
