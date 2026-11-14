# GECKO: Federated Continual Graph Learning

Code, dataset construction, method configurations, fixed data splits and trained
evaluation artifacts for reviewing and reproducing GECKO. This is release 0.4.2;
no private account, server, tracking service or Git history is needed.

```text
src/gecko/                  benchmark implementation and construction APIs
reproduce/
  configs/                  scenario and method YAML configurations
  splits/                   228 frozen stream cells, with tensor checksums
  artifacts/                trained reference weights for all eight stages
  environment.yml           CPU installation recipe
  construct.py              construct new scenario streams from public data
  run.py                    train on an existing fixed split
  evaluate.py               evaluate released weights without training
  check.py                  verify release, splits and trained tensors
  smoke.py                  small CPU construct/audit/train/report example
  package.py                export this source snapshot without Git history
  checksums.json            release-file integrity manifest
README.md                   this guide (the only Markdown file)
pyproject.toml, MANIFEST.in  installation and source-distribution metadata
LICENSE, NOTICE             code license and required third-party attribution
```

## Install and check

Run from the extracted source directory:

```bash
conda env create -f reproduce/environment.yml
conda activate gecko
python -m pip install --no-deps -e .
python reproduce/check.py --smoke
```

The smoke workflow uses synthetic data on CPU. For real-data loaders and native
backbones, add the matching graph dependencies to this CPU environment:

```bash
python -m pip install --only-binary=:all: 'torch-scatter==2.1.2+pt27cpu' \
  -f https://data.pyg.org/whl/torch-2.7.0+cpu.html
python -m pip install -e '.[graph,pyg,legacy-gem]'
```

The CPU recipe uses Python 3.10 and PyTorch 2.7.1+cpu. CUDA requires compatible
PyTorch, DGL and extension wheels; do not install CPU extension wheels into a CUDA
environment. The trained reference was checked with Python 3.11.15,
PyTorch 2.7.1+cu128, DGL 2.5, NumPy 1.26.4 and torch-scatter 2.1.2+pt27cu128.
The CPU recipe and the checked CUDA runtime are distinct environments; numerical
identity across every library/hardware combination is not guaranteed. Tracking
is disabled by the reproduction scripts.

## Restore predefined inputs

```bash
python -m gecko.data.predefined list
python -m gecko.data.predefined verify
python -m gecko.data.predefined materialize --id S5-s0-a100-synchronized \
  --output generated_streams/predefined
```

Each cell is one seed × allocation × order, shared by all compared learners.
The saved payloads contain client ownership, train/validation/test query IDs and
labels, class/domain masks, LP candidates, task orders and complete participation
traces. Import restores these assignments without sampling new splits. JSON and
compressed NumPy arrays are checked by content hash; no pickle is loaded.

| Family | Seeds | Allocation alpha | Orders | Clients | Cells |
| --- | --- | --- | --- | --- | --- |
| S1: Arxiv Task-IL | 0–2 | 0.1, 100 | Sync, Unsync | 10 | 12 |
| S2: Arxiv Class-IL | 0–2 | 0.1, 1, 10, 100 | Sync, Unsync | 10 | 24 |
| S3: Proteins Domain-IL | 0–2 | 0.1, 1, 10, 100 | Sync, Unsync | 10 | 24 |
| S4: Bitcoin Task-IL | 0–2 | 0.1, 100 | Sync, Unsync | 10 | 12 |
| S5: Bitcoin Class-IL | 0–2 | 0.1, 1, 10, 100 | Sync, Unsync | 10 | 24 |
| S6: Bitcoin Domain-IL | 0–2 | 0.1, 100 | Sync, Unsync | 10 | 12 |
| S7: legacy Facebook LP | 0–4 | 0.1, 1, 10, 100 | Sync, Unsync | 3 | 40 |
| Facebook clean LP | 0–4 | 0.1, 1, 10, 100 | Sync, Unsync | 3 | 40 |
| Wiki-CS LP | 0–4 | 0.1, 100 | Sync, Unsync | 10 | 20 |
| OGBL-Collab LP | 0–4 | 0.1, 100 | Sync, Unsync | 10 | 20 |

Unsync is serialized as `hard` or `binary_mismatch`; federated communication is
synchronous throughout. Base/Alloc./Order/Joint are (100, Sync)/(0.1, Sync)/
(100, Unsync)/(0.1, Unsync). S1–S3 have eight stages, S4–S5 three, S6 four and LP
two; the main budget is 50 rounds/stage and one local epoch. Participation is
5/10 clients for NC/LC and the new LP datasets, and 2/3 for Facebook LP.
Wiki-CS uses within-topic/across-topic domains; Collab uses first collaboration
through/after 2010. Both use 1,000 fixed evaluation negatives per client–task,
separately for validation and test, shared by methods.

Compact LC/LP derived graphs are included. NC requires the public raw dataset:

```bash
python -m gecko.data.predefined materialize --id S2-s0-a100-synchronized \
  --data-root data --output generated_streams/predefined
```

The loader downloads missing raw data to `--data-root` and checks raw and
reconstructed tensors against the frozen hashes. A mismatch stops import.
Dataset terms and citations remain those of the providers:
[OGB](https://ogb.stanford.edu/docs/nodeprop/),
[Bitcoin-OTC](https://snap.stanford.edu/data/soc-sign-bitcoinotc.html),
[GEMSEC Facebook](https://snap.stanford.edu/data/gemsec-Facebook.html),
and the Wiki-CS/Collab sources used by the included loaders.

## Evaluate pretrained weights

```bash
python reproduce/evaluate.py --list
python reproduce/evaluate.py --id S2-MOTION-Base-seed0 --data-root data \
  --device cuda:0 --output outputs/pretrained-evaluation
```

**Coverage:** one real trained reference run, native MOTION on S2 Base, seed 0,
with all eight stage checkpoints (400 training rounds in total). This is the
completed task-order robustness study's Sync reference, selected by its fixed
scenario/anchor/seed identity. It is not a collection of pretrained weights for
every paper or rebuttal run.

The export retains the exact shared evaluation parameters, client-local buffers,
classifier state and task exposure at each stage. Training replay memories,
optimizer/resume state and private machine paths are omitted. These are real
trained weights in checksummed NumPy tensors, not placeholder models or saved
scores alone. The inference export is not a training-resume checkpoint.

Evaluation performs no training. It recomputes all 360 eligible client–stage–task
accuracies and checks them against the archived run, then writes `per_cell.csv`
and `summary.json` with final accuracy, acquisition, AF and the maximum cell
error. Values are fractions; multiply by 100 for percentages/percentage points.
A mismatch beyond the declared tolerance exits with an error. On the checked
CUDA runtime all 360 cells matched exactly. CPU inference also runs, but the
checked CPU pass had a maximum cell difference of 0.001642 (0.1642 percentage
points); final accuracy and AF matched, while acquisition differed by 0.002664
percentage points. Therefore CPU inference does not pass the default strict
1e-6 comparison against the archived CUDA run. The script reports this rather
than silently relaxing the tolerance. Other methods/cells require training from
the fixed inputs below.

## Train or construct

Inspect a recipe before running it:

```bash
python reproduce/run.py --id S5-s0-a100-synchronized --method GEM --plan
python reproduce/run.py --id S5-s0-a100-synchronized --method GEM --device cpu \
  --checkpoint-dir outputs/S5-GEM-Base-seed0/checkpoints
python -m gecko.benchmarks.paper --stage plan
```

`run.py` restores the saved cell, selects the original scenario-specific method
configuration and uses the normal audited runner. It preserves native MOTION,
POWER and other method-specific aggregation. S5 is a full 150-round training run.
Results are written under `generated_streams/predefined/<split-id>/`; optional
checkpoints are written to the supplied directory. Read `--help` for arguments.

To construct new streams and run the paper's main recipes:

```bash
python reproduce/construct.py --scenarios S2 \
  --data-root data --output outputs/new
python -m gecko.benchmarks.paper --stage run --scenarios S2 --methods GEM \
  --device cpu --output outputs/new
```

Use predefined splits for exact saved assignments. New construction follows the
recipe but may differ with raw-data/partition-library versions. Original S3 is
construction-only in the paper catalog. Eligibility checks remain enforced.
Method variants, local-only/FedProx comparisons and additional diagnostics must
use their explicitly matching configurations; the main runner is not a recipe
for every appendix or later rebuttal experiment.

## Scope and interpretation

The original source release passed 946 tests; packaging tests do not certify
all manuscript numbers. This minimal distribution retains runnable verification
scripts instead of the development test suite. Saved inputs match the audited
primary S2/S5/S7 provenance. Independent end-to-end numerical reproduction of
all reported tables has not been established, and weights for all runs are not
included.

Changes relative to the original supplementary source are disclosed here:

- Fixed split tensors and installation metadata are now included. Historical
  and corrected protocols have distinct configurations and split IDs.
- Facebook clean fixes category-local ID collisions and uses strict-local base
  degree features; it is a new protocol, not a feature-only intervention. Legacy
  Facebook uses global-derived degree features that include held-out edges.
- Revised LP sampling can reserve evaluation negatives before training negatives.
  Preserve the constructor version and fixed candidate population.
- GraphKeeper-LP currently consolidates the federated backbone at stage boundaries
  (v4 versus the submitted v3). Ridge is the default; the optional dot-product
  head is a diagnostic variant, not an unchanged submitted algorithm.
- Two original S6 cells (seed 1, alpha 0.1, both orders) fail validation support
  for task 3/class 0 and remain blocked. The corresponding three-seed Table 22
  summaries cannot be certified as strict reproductions under these checks.
- Original S3 has undefined local AUROC pairs and construction-only coverage.
  Later S3 learner protocols are separate experiments. Undefined metrics are not
  replaced with zero. The original GEM qpth solver is not silently replaced by
  BVLS; native MOTION is not replaced by a GCN-only approximation.

Forgetting is computed only after task exposure. NC/LC final performance uses
valid client–task macro aggregation; LP final Hits@50 is positive-query weighted
and AF is client–task macro. LP cutoff ties are pessimistic. Compute paired seed
contrasts first, then mean and sample SD; multiple order profiles are not extra
independent seeds. Preserve failures and negative results when reporting.

## Share the source snapshot

```bash
python reproduce/package.py --output /tmp/gecko-reviewer.zip
```

The archive contains only the distribution shown above; it excludes Git history,
logs, local environments, generated runs and development reports. Use the
extracted snapshot for anonymous hosting: the existing repository's commit
history is not anonymized. Retain `LICENSE` and `NOTICE`, including third-party
attribution. No other Markdown files are distributed.
