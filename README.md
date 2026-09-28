# GECKO

## Installation

```sh
conda env create -f environment.yml
conda activate gecko
python -m pip install --only-binary=:all: 'torch-scatter==2.1.2+pt27cpu' \
    -f https://data.pyg.org/whl/torch-2.7.0+cpu.html
python -m pip install -e '.[graph,pyg]'
```

## Construction

```sh
python -m gecko.benchmarks.paper --stage construct \
    --scenarios S2 \
    --data-root /path/to/datasets \
    --output /path/to/outputs
```

## Training and Evaluation

```sh
python -m gecko.benchmarks.paper --stage run \
    --scenarios S2 --methods GEM \
    --device cpu \
    --output /path/to/outputs
```

## Tests

```sh
python -m gecko.benchmarks.paper --stage test --scenarios S2 --methods GEM
```

See `python -m gecko.benchmarks.paper --help` for available options.
