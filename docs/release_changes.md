# Release 0.4.1: review snapshot and scientific changes

The originally submitted supplementary ZIP is the archival submission artifact.
Its SHA-256 is
`f7233e6673a5e65b4697ff44e29c85aba893733fd5a5b68fc255365354f80227`.
The current release does not claim to be that unchanged ZIP or to reproduce all
submitted scores. Both the historical protocol and the opt-in corrected LP
protocol are explicitly identifiable in configs and the split catalog.

## Packaging and reproducibility

- Include the package installation metadata, documented CPU environment, full
  source/tests/configurations, and fixed split artifacts. The submitted ZIP did
  not include predefined split tensors or packaging metadata for its editable
  installation command.
- Add JSON/NumPy split interchange and checksummed restoration, preserving the
  actual client ownership, query labels/splits/candidates, task arrays, masks,
  and participation traces. Raw NC graph/features are reconstructed and checked
  against their exact tensor hashes. Original stream fingerprints remain
  provenance; schema-2 imported streams receive a new container identity.
- Remove machine-transfer manifests, server-specific launchers, internal paths,
  experiment logs and archived baselines from the release snapshot. They remain
  in historical version control; the anonymous export excludes Git entirely.
- Retain third-party credits and the normal stream and method eligibility gates.
  Presence of an implementation or a passing source test is not promotion to
  benchmark-eligible status or proof of reported numerical accuracy.

## Changes that can affect scientific results

1. **Facebook clean input protocol (opt-in).** Correct category-local node-ID
   collisions and derive degree features only from each client's fixed base
   graph. Ownership, splits and candidates belong to a new protocol; clean LP
   results are not a feature-only intervention on the submitted legacy inputs.
   Legacy S7 assignments remain separately available.
2. **LP negative sampling (opt-in).** The revised protocol can allocate fixed
   evaluation negatives before training negatives. The manifest and constructor
   version identify this setting. Do not mix candidate sets across protocols.
3. **GraphKeeper-LP.** The current implementation consolidates the federated
   backbone at stage boundaries (v4 versus the submitted v3 implementation).
   An explicit `prediction_head: dot_product` diagnostic variant bypasses the
   recursive ridge head. Ridge remains the default; dot-product results must be
   labeled as that variant, not presented as an unchanged submitted algorithm.
4. **Additional LP datasets.** Wiki-CS and OGBL-Collab have explicit two-domain,
   ten-client configurations and fixed anchor splits. These are GECKO continual
   LP protocols, not claims of reproducing the official OGB leaderboard or the
   original Wiki-CS node-classification protocol.

The original paper runner retains the submission's scenario/method catalog.
Later S3 learner runs, LC optimization interventions, hyperparameter studies,
randomized-order extensions and solver/backend experiments must be described
with their own protocol and method settings. This snapshot is the complete
benchmark source and fixed primary split release, not a dump of every internal
campaign, checkpoint, or result. It does not automatically reproduce every
rebuttal diagnostic. In particular, the original qpth GEM path is not silently
replaced by BVLS, and native MOTION is not replaced by a GCN-only learner.

## Known unresolved manuscript discrepancy

The two original S6 streams at seed 1 / alpha 0.1 (Sync and Unsync) fail
validation support for task 3/class 0. Their assignments are included for
inspection and remain ineligible under the enforced audit. Three-seed Table 22
summaries containing those inputs cannot be certified as strict reproductions
by this release. S3's original catalog is construction-only; undefined local
ROC-AUC pairs are not filled with zero or silently converted to valid runs.

Describe these scientific changes when linking an updated artifact to reviewers;
keep historical and corrected tables separate.
