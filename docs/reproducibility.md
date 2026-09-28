# Reproduction

## Scope

This source archive contains implementations, configurations, and scientific
tests. It does not contain datasets, frozen experiment payloads, or a claim
that all manuscript numbers have been reproduced. `examples/benchmark.py`
exposes the seven scenarios in manuscript Table 2 through the existing
construction and training APIs. It does not reimplement the algorithms.

The default recipes cover the main 50-round, one-local-epoch panels and
their documented matched anchors. They do not automatically reproduce every
appendix diagnostic, training-rule sensitivity, or intervention experiment.
See [the table-level coverage map](paper_alignment.md#paper-panel-coverage).

## Environment

`environment.yml` is a pinned CPU recipe for Linux x86-64: Python 3.10.20,
PyTorch 2.7.1+cpu, NumPy 1.26.4, SciPy 1.15.3, NetworkX 3.4.2, PyYAML 6.0.2,
and pytest 8.4.1. This package set was recreated in a fresh virtual
environment and used for the synthetic workflow and selected correctness
tests. This is not a claim that a fresh Conda solver run, CUDA installation,
or full paper experiment was verified. Installation requires network access.

For an explicit virtual-environment installation with Python 3.10 available:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled
python -m pip install 'torch==2.7.1+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'numpy==1.26.4' 'scipy==1.15.3' 'networkx==3.4.2' 'PyYAML==6.0.2' 'pytest==8.4.1'
python -m pip install --no-deps -e .
python examples/synthetic.py --output outputs/synthetic
```

With that CPU environment active, the following graph installation was also
successfully recreated:

```bash
python -m pip install --only-binary=:all: 'torch-scatter==2.1.2+pt27cpu' \
  -f https://data.pyg.org/whl/torch-2.7.0+cpu.html
python -m pip install -e '.[graph,pyg]'
```

The original real-data loaders/models require `torch-scatter`; DSLR also
uses the `pyg` extra. Legacy GEM paths use `.[legacy-gem]`. The `graph` extra
includes PyMETIS for real-data partitioning. A deterministic BFS fallback
can support synthetic construction without PyMETIS, but it is not a
substitute for the real-data partition recipe.

Changing PyTorch or using CUDA requires a mutually compatible
PyTorch/DGL/extension matrix. Consult the
[official PyG installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
for matching extension wheels. Record the environment, GPU/driver, CPU thread
settings, and partition backend with results; the CPU environment does not
reconstruct an earlier CUDA runtime. Install `.[tracking]` only to opt in to
W&B. The examples disable tracking and require no credentials.

Training and checkpoint resume require no Git checkout, source manifest,
source checksum, or source commit. An optional ZIP checksum verifies a
downloaded archive only; it does not determine whether training can run.

## Data

| Dataset | Provider | Paper scenarios |
| --- | --- | --- |
| OGBN-Arxiv | [Open Graph Benchmark](https://ogb.stanford.edu/docs/nodeprop/) | S1, S2 |
| OGBN-Proteins | [Open Graph Benchmark](https://ogb.stanford.edu/docs/nodeprop/) | S3 |
| Bitcoin-OTC | [SNAP](https://snap.stanford.edu/data/soc-sign-bitcoinotc.html) | S4, S5, S6 |
| GEMSEC Facebook | [SNAP](https://snap.stanford.edu/data/gemsec-Facebook.html) | S7 |

Upstream terms and dataset citations apply; the source license does not
relicense datasets. Some loaders retrieve published BeGin split metadata.
Those upstream links are functional data sources. Raw and processed data
are intentionally absent. Use `--data-root data` or another writable cache
directory. Resolved experiment configurations belong under `--output`, not
inside the source package. Preserve the identities of downloaded inputs
when reporting a reconstruction.

## Runner stages and defaults

```bash
python examples/benchmark.py --help
python examples/benchmark.py --stage plan
python examples/benchmark.py --stage plan --scenarios S2 --methods GEM TWP \
  --allocation-alpha 0.1 100 --task-order synchronized --seeds 0 1 2
```

`plan` resolves recipes without downloading data or running learners. Other
stages require an explicit `--scenarios` selection. `construct` builds the
fixed streams; `run` audits the selected inputs and launches compatible
methods using each finalized stream's actual configuration. `test` runs
support-contract and relevant behavioral checks, not a numerical paper
reproduction. Use the same output directory and cell-selection arguments
at construction and training. Tracking is disabled by default.

The plan also exposes each method configuration's existing `support_status`
and `benchmark_eligible` fields. Several implementations remain marked
`implemented_unverified`; the runner does not upgrade those labels. An
audited eligible stream and an executable method are not, by themselves,
evidence that that method's results are eligible for benchmark rankings.
Report stream validity, method support, and completed numerical verification
separately.

The default grids are:

| Scenarios | Allocation alpha | Task order | Seeds | Clients / participants per round |
| --- | --- | --- | --- | --- |
| S1, S4, S6 | 0.1, 100 | Both | 0, 1, 2 | 10 / 5 |
| S2, S5 | 0.1, 1, 10, 100 | Both | 0, 1, 2 | 10 / 5 |
| S3, construction only | 0.1, 1, 10, 100 | Both | 0, 1, 2 | 10 / 5 |
| S7 | 0.1, 1, 10, 100 | Both | 0, 1, 2, 3, 4 | 3 / 2 |

Here “both” means synchronized and unsynchronized. The manuscript states
three paired NC/LC seeds and five paired LP seeds (Appendix C.1, page 19);
the runner uses the explicit LP seed IDs 0–4 as its reproducible convention.
S3's construction grid does not imply a learner panel. All main learner
recipes use 50 communication rounds per stage and one local epoch, with
every client participating at least once per stage.

The four matched anchors (Appendix B.1, equation 8, page 15) are:

| Anchor | Allocation alpha | Task order |
| --- | --- | --- |
| Base | 100 | Synchronized |
| Alloc. | 0.1 | Synchronized |
| Order | 100 | Unsynchronized |
| Joint | 0.1 | Unsynchronized |

### Default methods

| Scenario | Method panel |
| --- | --- |
| S1 | GEM, TWP, FedGTA, FedDC, POWER, MOTION |
| S2 | Bare, CaT, DSLR, FedDC, FedGTA, FedPUB, GEM, MOTION, POWER, SSM, TWP; plus synchronized-only FedFST |
| S3 | None: construction only |
| S4 | GEM, TWP |
| S5 | Bare, CaT, SSM, TWP, GEM |
| S6 | GEM, TWP |
| S7 | Bare, EWC, LwF, GraphKeeper |

FedFST is an additional S2 synchronized-only comparison over all four
allocation values, not part of the eleven-method full-grid summary. The
S2 default schedules that supplement only on synchronized cells. To select
it alone:

```bash
python examples/benchmark.py --stage plan --scenarios S2 --methods FedFST --task-order synchronized
```

FedDC, FedFST, FedGTA, FedPUB, MOTION, and POWER retain their native
federated mechanisms. Other main-panel methods use FedAvg, including CaT,
DSLR, and every S7 method. The general method registry can expose other
training rules; its mere availability does not make them paper defaults.
The default backbone is the three-layer, width-256 GCN with Adam learning
rate 0.01 and no weight decay. FedFST instead uses its configured two-layer,
one-head, width-64 GAT. FedPUB performs two optimizer updates per local
epoch. Communication budgets are matched, not total method compute
(Appendix C, Tables 11–13, pages 19–20).

### Construction lineage and output

NC/LC reconstruction follows the exact-Dirichlet LPT v1 construction and
the existing participation derivation to the 50x1 main budget. LP uses its
separate domain-incremental construction contract; it is not an NC/LC LPT
stream with the dataset name replaced. Scenario defaults also override
generic configuration values where needed, notably S7's three clients,
two participants, and five seeds. The runner's plan is the place to inspect
these resolved choices before allocating a large experiment.

S7 construction requires at least two allocation values so that its
cross-allocation manipulation audit is meaningful. Prefer the default
four-value grid; a later run may select one value from an already audited
grid. A source-construction failure is not silently removed from the plan.

Each scenario records its finalized grid at
`<output>/S1/stream_manifest.json`, with `S1` replaced by the selected ID.
NC/LC keep the source `streams_10x1` and derived `streams_50x1` directories;
LP constructs `streams_50x1` directly. These are generated scientific-data
records, not source manifests that the user must prepare before training.

Fresh exact-allocation stream directories use descriptive components such
as `allocation-alpha-0.1/task-order-unsynchronized`. Historical serialized
schema names can remain in the surrounding path. Do not locate a stream
by guessing a legacy `spatial-mild` directory: use the runner's recorded
paths and each stream's `config.yaml`. Existing metadata, scientific
identifiers, and old finalized artifacts are not rewritten merely to
change display names. See [compatibility boundaries](paper_alignment.md#naming-and-compatibility).

Changing dataset versions, partitioning runtime, thread behavior, or
serialization can produce different payloads. These commands reconstruct
the stated recipes; they do not restore byte-identical historical artifacts
or validate published numerical results. Use a new output directory for a
changed stream-defining recipe. Use the same finalized streams for compared
methods, and report completed, failed, and excluded cells separately.

## Scientific boundaries and known discrepancy

The graph is static, ownership is persistent, and each client receives an
induced local graph. Query shards, task IDs, splits, negatives, and
participation are fixed before training. Method/model selection must not
regenerate them. Runtime views exclude remote features, cross-client edges,
validation/test labels, future-task training labels, complete ownership maps,
and LP evaluation positives as message-passing context. Reverse arcs of an
undirected logical edge stay together; original directed Bitcoin
interactions, including reciprocal observations, remain distinct.

The offline constructor is trusted preprocessing, not a label-blind client
algorithm. The paper describes allocation using training supervision and
support checks, and discloses global-derived structural features (Appendix A,
pages 13–14). Bitcoin uses global in/out-degree features. Facebook degree
features are derived from the full positive graph, including held-out
positive edges. Thus “no evaluation positives in runtime context” is not
equivalent to “all structural features exclude evaluation edges.” Strict-local
execution is not differential privacy or secure aggregation.

NC/LC allocation may change retained query populations; in particular the
LC allocation comparison is not a fixed-evaluation-population intervention.
LP instead fixes ownership, its base graph, and evaluation queries across
allocation values, varying selected training positives and context. LP
negatives exclude all known positives in both orientations and self-pairs.
Its evaluation context is the base graph plus the queried task's training
positives, with reverse arcs, not future-task positives. Preserve these
distinct semantics when interpreting alpha.

S3 is explicitly construction-only in the manuscript. Earlier frozen
NC-Domain shards include undefined local ROC-AUC; this archive does not
turn those into learner results or silently fill missing metrics. The
separate complete-metric protocol is a different experiment, not an
equivalent replacement for S3.

For S6, the frozen seed-1 / alpha-0.1 inputs under both task orders fail
validation-query support for task 3/class 0 and are marked ineligible.
Nevertheless, manuscript Table 22 (page 25) reports the corresponding
three-seed anchor summaries. Those reported summaries therefore cannot be
strictly reproduced from the blocked inputs under the enforced eligibility
policy. The runner refuses ineligible streams; this packaging work does not
change the exclusion rule, certify those scores, or silently omit failed
seeds. Resolving the discrepancy requires an explicitly documented
scientific correction, not a diagnostic bypass.

## Metrics and verification

NC/LC summaries use uniform valid client-task aggregation. LP final
performance is positive-count weighted, while forgetting remains uniformly
aggregated; its pessimistic Hits@50 convention treats cutoff ties as misses.
Forgetting excludes stages before task exposure. Compute seed-level
summaries first, then their mean and sample standard deviation (Section 4,
page 5; Appendix C.2, page 19). Do not pool all cells as if they were
independent seeds or silently substitute zero for undefined metrics.

After extraction, run the synthetic example and selected scenario tests,
then the full included suite with required optional dependencies installed.
Record skips, failures, hardware, and commands honestly. Passing source tests
or an audit is not equivalent to reproducing a benchmark table.

Finalized stream metadata and tensor checksums protect scientific data from
corruption and are generated automatically by construction. They are
distinct from source-code checks and do not require Git. Keep finalized
scientific streams immutable; preserve resolved configurations, data
identities, environment details, and per-seed results with experiments.
