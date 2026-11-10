# GECKO: A Diagnostic Benchmark for Federated Continual Graph Learning

This reviewer release contains the benchmark implementation, construction and
training entry points, method configurations, correctness tests, and **actual
predefined client/data splits**. No account, API key, private server, or Git
history is required. Tracking is disabled in the examples.

The original supplementary ZIP contained source and configurations. This release
adds portable fixed split artifacts and incorporates explicitly documented
post-submission changes. See [release changes](docs/release_changes.md) before
comparing a new run with a submitted-paper number.

## Install from the extracted source directory

```bash
conda env create -f environment.yml
conda activate gecko
python -m pip install --no-deps -e .
python examples/synthetic.py --output outputs/synthetic
```

The synthetic example requires only the core CPU environment. To use the
real-data loaders and native method backbones in that environment:

```bash
python -m pip install --only-binary=:all: 'torch-scatter==2.1.2+pt27cpu' \
  -f https://data.pyg.org/whl/torch-2.7.0+cpu.html
python -m pip install -e '.[graph,pyg,legacy-gem]'
```

`environment.yml` is the pinned CPU recipe. CUDA wheels require a compatible
PyTorch/DGL/extension installation; [reproduction details](docs/reproducibility.md)
and [validation evidence](docs/release_validation.md) distinguish the tested
runtime from the CPU installation recipe. Run from this source directory so
that configs, tests, and split artifacts remain available.

## Use the predefined splits

```bash
python -m gecko.data.predefined verify
python -m gecko.data.predefined list
python tools/run_predefined.py --id S5-s0-a100-synchronized --method GEM --plan
python tools/run_predefined.py --id S5-s0-a100-synchronized --method GEM --device cpu
```

The last command reconstructs the **saved** assignments and launches the original
method configuration through the normal audited runner. It does not sample a new
partition, query split, task order, or participation trace. S5 here is a complete
150-round run, not a quick smoke test. Use `--device cuda:0` only with a compatible
CUDA environment. Artifacts go to `generated_streams/predefined/<split-id>/`.

For NC, supply/download the public raw dataset with `--data-root data`; every raw
and reconstructed graph tensor must match the frozen content hashes. Compact
LC/LP derived graph and query tensors are included. Dataset terms remain those
of the original providers. [Split documentation](splits/README.md) lists exact
coverage, format, provenance, and how to inspect train/validation/test membership.
`verify` checks stored payloads; `materialize` additionally checks external raw
and reconstructed tensors. Neither command certifies published accuracy.

```bash
python -m gecko.data.predefined materialize --id S2-s0-a100-synchronized \
  --data-root data --output generated_streams/predefined
```

## Construct new streams or inspect the submitted-paper recipes

```bash
python -m gecko.benchmarks.paper --stage plan
python -m gecko.benchmarks.paper --stage construct --scenarios S2 --data-root data --output outputs/new
python -m gecko.benchmarks.paper --stage run --scenarios S2 --methods GEM --device cpu --output outputs/new
python -m gecko.benchmarks.paper --stage test --scenarios S2 --methods GEM
python -m pytest -q
```

Fresh construction is useful for new experiments; use the predefined artifacts
for exact historical assignments. The paper catalog retains the submission's
seven-scenario coverage, budgets and method-specific architectures/aggregation.
It does not silently relabel later extensions as original results. Ineligible
streams remain blocked. See [paper alignment](docs/paper_alignment.md),
[known reproducibility limitations](docs/reproducibility.md#scientific-boundaries-and-known-discrepancy),
and the [current LP protocol](docs/lp_clean_v2.md).

## Share the reviewer snapshot

```bash
python tools/export_reviewer_archive.py --output /tmp/gecko-reviewer.zip
```

This exports source, configs, tests, docs and predefined splits without `.git`,
commit metadata, experiments, logs, credentials, checkpoints, or local caches.
Use the extracted snapshot for anonymous hosting. Preserve `LICENSE.md` and
`NOTICE`, including third-party attribution. Sharing this snapshot does not
publish the existing repository's history.
