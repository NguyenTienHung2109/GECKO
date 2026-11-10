# Reviewer release validation

Validation for release 0.4.1 separates source tests, fixed-input restoration and
numerical reproduction. Source checks do not certify the submitted tables.

## Tests and runtime

The full included suite passed: **946 tests**, with 863 third-party/runtime
warnings (no failed tests). This includes all seven synthetic scenario contracts,
stream and method support checks, checkpoint/resume, privacy boundaries, metrics,
GraphKeeper ridge/dot-product behavior, fixed-split round trips, corrupted-input
rejection, packed mapping preservation, and source-path portability.

The validation environment was Linux, Python 3.11.15, PyTorch 2.7.1+cu128,
DGL 2.5, NumPy 1.26.4, SciPy 1.11.4, NetworkX 3.6.1, PyYAML 6.0.1,
pytest 9.1.1, OGB 1.3.6, PyMETIS 2023.1.1, qpth 0.0.18,
torch-scatter 2.1.2+pt27cu128 and torch-geometric 2.4.0. CPU thread counts were
limited for testing. CUDA-enabled libraries do not imply that a new full GPU
campaign or CUDA/backend numerical equivalence was verified in this release.

The pinned CPU recipe in `environment.yml` is retained separately; the checks
above did not rerun the Conda solver or certify every optional-library version
combination. Test the synthetic example before launching real-data workloads.

## Fixed inputs

All original source tensor files were checked against their recorded SHA-256
before extraction; 1,140 sidecar checksums also matched. The catalog fingerprints
match all 544 main-panel result records in the original S2/S5/S7 provenance.
NC induced graph reconstruction was checked against the
original client graph tensors during export. Public compressed payloads are
verified independently by `python -m gecko.data.predefined verify`.

Real-data checks include Arxiv and Proteins raw loader tensor parity, full
Arxiv/Proteins split imports, and Bitcoin Class-IL and Domain-IL import through
the normal schema-2 artifact store and public loader. Bitcoin Class-IL remains
eligible; the known ineligible seed-1/alpha-0.1 Domain-IL stream remains blocked.
Offline imports of legacy Facebook, clean Facebook, Wiki-CS and Collab LP
also pass the public loader/audit. No split sampler or eligibility bypass is
used by the importer.

See `release_validation.json` for the final catalog/packaging check counts.
The fixed-input importer does not prove numerical accuracy equality across
software, hardware, solver or algorithm versions. Historical and corrected LP
inputs and the v4/dot-product GraphKeeper changes remain explicitly distinct.

## Distribution checks

The exported source ZIP omits Git history and archive comments, machine-transfer
bundles, logs, credentials, model checkpoints, generated output stores, compiled
Python files and local environments. It retains license notices, third-party
credits, construction code, method configs, tests and the predefined data.
The `SHA256SUMS` file covers the release snapshot (excluding itself).

The source ZIP was extracted into an independent directory without `.git` and
installed in a new virtual environment using preinstalled dependency libraries.
The synthetic construct/train/report workflow, paper plan, fixed-split plan,
offline S5 materialization and 12 additional split/portability tests passed there.
This verifies source distribution portability; it is not a fresh online
dependency-resolution test. The final snapshot scan found no submitting-user
names, affiliations, private paths, identifying repository URLs, or credentials
in the scanned text/cell manifests. Third-party attributions were retained.
