# Paper alignment and naming

Page and table numbers below refer to *GECKO: A Diagnostic Benchmark for
Federated Continual Learning on a Shared Graph*, the accompanying 32-page
manuscript. They identify intended experimental recipes, not a certificate
that the numerical tables have been rerun.

## Scenario map

Table 2 (page 4) defines the seven public scenario IDs. The scenario files
below reside in `configs/gecko_v1/scenarios/`; the paper runner resolves
their budget, allocation, order, method, and participation overrides.

| Paper ID | Problem / regime | Dataset | Tasks and outputs | Scenario configuration |
| --- | --- | --- | --- | --- |
| S1 | Node classification / Task-IL | OGBN-Arxiv | 8 tasks, 5 classes per task | `nc_task_ogbn_arxiv.yaml` |
| S2 | Node classification / Class-IL | OGBN-Arxiv | Same 8 tasks; 40 classes overall | `nc_class_ogbn_arxiv.yaml` |
| S3 | Node classification / Domain-IL | OGBN-Proteins | 8 species domains; 112 binary outputs | `nc_domain_ogbn_proteins.yaml` |
| S4 | Link classification / Task-IL | Bitcoin-OTC | 3 tasks, 2 labels per task | `lc_task_bitcoin.yaml` |
| S5 | Link classification / Class-IL | Bitcoin-OTC | Same 3 tasks; 6 task/class labels overall | `lc_class_bitcoin.yaml` |
| S6 | Link classification / Domain-IL | Bitcoin-OTC | 4 degree-based domains; 7 labels | `lc_domain_bitcoin.yaml` |
| S7 | Link prediction / Domain-IL | GEMSEC Facebook | 2 source groups | `lp_domain_facebook.yaml` |

Task-IL uses the known global task to mask outputs; Class-IL permits each
client's locally seen classes; Domain-IL retains a fixed output space and
does not reveal a domain identity at prediction time. S3 provides
construction-only scenario definition and validation, not a learner panel.

### Dataset details that affect interpretation

Appendix A and Table 7 (pages 13–15) specify more than dataset names:

- Arxiv has 169,343 nodes, 40 labels, 128-dimensional features, and the
  official 90,941 / 29,799 / 48,603 train/validation/test split. Its eight
  fixed five-label tasks are shared between S1 and S2. Reverse arcs and
  self-loops are constructed as specified, without using split year or
  labels as node features.
- Proteins has 132,534 nodes, 112 binary targets, and eight edge-feature
  channels. The paper uses a custom within-domain split, not the official
  OGB evaluation protocol. Node features average outgoing locally owned
  edge features. Species domains and binary output labels are different
  concepts.
- Bitcoin contains 35,592 directed interactions on 6,005 nodes. S4/S5 use
  three task pairs: ratings `{3}` versus `{4}`, `{-10}` versus `{-9,...,-1}`,
  and `{6,...,10}` versus `{5}`. Their 8,002-item training pool is not the
  selected training budget: the paper selects 398 training queries. The
  seventh rating group is absent from these task/class targets but included
  in S6. S6 uses all 35,592 interactions and degree-based domains whose
  quartiles are defined using training data.
- Facebook has eight source categories but two task groups: Artist versus
  the other seven. The 1,380,293 positive rows span 134,826 nodes because
  seven node indices are shared across category boundaries; the graph is
  not a disjoint union of categories. The LP protocol uses three clients,
  a roughly 10% base graph, fixed evaluation queries, and task-specific
  retained training positives. It is not an LC protocol with binary labels.

The supplied configurations and constructor determine actual queries and
support checks. These descriptions do not authorize bypassing those checks.

## Experimental axes and budgets

The paper's axes are **client allocation** and **task order**. Allocation
alpha takes values `{0.1, 1, 10, 100}`. It is not interchangeable with a
universal label-skew index, a legacy profile label, or “difficulty.” In
NC/LC, allocation changes ownership, local topology, and retained queries.
In LP, ownership, base graph, and evaluation are fixed across alpha; selected
training positives and their context change instead (Appendix B.1, page 15).

Synchronized order gives clients the same task order. Unsynchronized order
uses permutations within the two four-task NC blocks, within the three/four
LC tasks, or opposite orders for the two LP tasks. Participation is fixed
across paired order comparisons. A historical partial-order `mild` profile
is not the paper's unsynchronized condition.

The main grid is four allocation values by two task orders. Matched
anchors are Base `(100, synchronized)`, Alloc. `(0.1, synchronized)`,
Order `(100, unsynchronized)`, and Joint `(0.1, unsynchronized)`.

Table 11 (page 19) and Appendix B.4 (page 17) specify 50 rounds per stage,
one local epoch, three paired NC/LC seeds, and five paired LP seeds. NC/LC
use 10 clients with 5 participants per round; LP uses 3 clients with 2
participants. Every client participates at least once in each stage.
Stages advance synchronously even when clients encounter different global
task IDs. Matching communication rounds does not equalize optimizer steps,
replay work, auxiliary reconstruction, or total computation.

## Paper panel coverage

| Panel | Scenarios / cells | Methods | Runner status |
| --- | --- | --- | --- |
| Main NC summary, Table 3 (p. 6); Tables 14–15 (pp. 21–22) | S2, full grid | Bare, CaT, DSLR, FedDC, FedGTA, FedPUB, GEM, MOTION, POWER, SSM, TWP | Default S2 panel |
| Synchronized NC extension, Tables 14–15 | S2, all four alpha values, synchronized only | FedFST | Included as a synchronized-only supplement in the S2 default; may also be selected explicitly |
| LC grid, Tables 16–17 (pp. 22–23) | S5, full grid | Bare, CaT, SSM, TWP, GEM | Default S5 panel |
| LP grid, Tables 18–19 (p. 23) | S7, full grid | Bare, EWC, LwF, GraphKeeper | Default S7 panel; all FedAvg |
| Matched NC regimes, Table 20 (p. 24) | S1/S2, four anchors | GEM, TWP, FedGTA, FedDC, POWER, MOTION | Default S1; select matching S2 subset |
| Matched LC regimes, Table 21 (p. 24) | S4/S5, four anchors | GEM, TWP | Default S4; select matching S5 subset |
| LC Domain-IL extension, Table 22 (p. 25) | S6, four anchors | GEM, TWP | Default S6 recipe; blocked-input discrepancy below |
| Construction-only scenario, Table 2 (p. 4) | S3 | No learner panel | Construction and contract tests only |

S2 and S5 contain their anchor subsets, but a matched-regime comparison must
select the same methods, seeds, allocation values, and orders on each side.
For example:

```bash
python examples/benchmark.py --stage plan --scenarios S1 S2 \
  --methods GEM TWP FedGTA FedDC POWER MOTION --allocation-alpha 0.1 100
python examples/benchmark.py --stage plan --scenarios S4 S5 \
  --methods GEM TWP --allocation-alpha 0.1 100
```

### Method and backbone choices

Appendix C, Tables 12–13 (page 20), distinguishes a method's native
federated mechanism from a continual learner attached to FedAvg. FedDC,
FedFST, FedGTA, FedPUB, MOTION, and POWER keep their native mechanisms.
Bare, CaT, DSLR, GEM, SSM, TWP, EWC, LwF, and GraphKeeper use FedAvg in the
main panels where they appear. Registry entries for Local-only or FedProx do
not override this table-specific choice.

Most methods use a three-layer, hidden-width-256 GCN, Adam learning rate
0.01, and zero weight decay. FedFST uses a two-layer, single-head,
hidden-width-64 GAT with dropout 0.5 and its generator/distillation
optimizers. FedPUB takes two optimizer steps per local epoch. Important
method settings include GEM's 100-query memory, SSM's `[10, 25]` sampling,
TWP's `1e4` importance coefficients, EWC/LwF regularization weight 1,
and GraphKeeper's rank-16 adaptation. The resolved method configurations,
not a common-name registry entry alone, specify these choices. Memory-capped
methods and auxiliary-compute methods must not be described as having
identical total resources merely because they share the 50x1 schedule.

### Panels not covered by a default main-grid run

- Training-rule sensitivity (Tables 26–30, pages 27–30) uses a separate
  10-round, five-local-update budget and compares Local-only, FedAvg, and
  FedProx with coefficient 0.01. NC uses CaT/GEM/SSM/TWP at Base/Alloc.;
  LC uses the same four methods at all four anchors; LP uses
  EWC/LwF/GraphKeeper at Base/Alloc. These are not the 50x1 main scores.
- Constructor sensitivity (Table 31, page 30) compares node-wise and
  topology-aware construction for synchronized S2 at alpha 0.1/100 using
  CaT, DSLR, GEM, SSM, and TWP. It requires distinct constructor settings.
- Prior, replay, and all-task controls (Table 4, page 8; Table 32, page 31)
  use S2/S5 Base and Joint. All-task training sees more examples despite
  matching update counts, so it is not a compute-matched upper bound.
- Cross-client support interventions (Table 5, page 9; Table 33, page 32)
  use balanced participation and checkpoint-specific Drop/Other task
  interventions. Running the ordinary main panel is not this experiment.

The runner's `test` stage checks implementations and contracts. It does not
produce any of these tables or certify their scientific conclusions.

## Naming and compatibility

Public names follow the manuscript while existing scientific objects remain
readable. Renaming a command or display label is separate from regenerating
a stream, changing a protocol, or rewriting an immutable artifact.

| Historical name or entry point | Public name / interpretation | Compatibility boundary |
| --- | --- | --- |
| Historical `begin`, `testbench`, or `uefa` package/CLI branding | Installed package and CLI: `gecko`; examples: `python -m gecko` | No old top-level namespace shim is promised; upstream BeGin attribution remains |
| Six textual scopes `nc_task`, `nc_class`, `nc_domain`, `lc_task`, `lc_class`, `lc_domain` | S1, S2, S3, S4, S5, S6 respectively; LP domain is S7 | Internal problem/regime strings and scenario filenames remain readable |
| “Spatial heterogeneity” or a generic spatial profile | **Client allocation** | Interpret the actual constructor; no universal difficulty ordering is implied |
| `--dirichlet-alpha` | `--allocation-alpha` | Legacy flag is an accepted alias; both refer to the numeric parameter |
| `partition.dirichlet_alpha` | `partition.allocation_alpha` in public configuration input | Old serialized field remains readable; alias does not change scientific values |
| “Temporal heterogeneity” or “order profile” | **Task order** | Global task IDs remain immutable; local order does not redefine a task |
| `--order-profile` | `--task-order synchronized\|unsynchronized` | Legacy profile interface remains accepted for compatibility |
| Historical `synchronized` order | `synchronized` | Same order condition, not a new protocol |
| Historical NC/LC `hard` or two-task LP `binary_mismatch` | `unsynchronized` | Alias selects the scenario-specific paper permutation; it does not make different task spaces equivalent |
| Historical order `mild` or `unconstrained` | Explicit legacy profiles, not a paper-order label | `mild` is not silently mapped to paper unsynchronized |
| `order.profile` | Public `order.task_order` input alias | Historical stored profiles remain readable and are not rewritten |
| Metadata such as `alpha_dirichlet`, `spatial_profile_label`, and `order_profile` | Explicit `allocation_alpha`, `allocation_control`, `allocation_label`, `task_order`, and `scenario_id` fields in new records | Public descriptive fields are added alongside historical fields; synthetic data has no paper scenario ID |
| Historical allocation `easy` / `mild` / `hard` labels | Numeric `allocation-alpha-<value>` only for exact-Dirichlet recipes | Never infer alpha from a heuristic label; heuristic and exact constructors are distinct |
| Exact-allocation output components `spatial-*` / `order-*` | `allocation-alpha-0.1/task-order-unsynchronized`, for example | New descriptive directories; old finalized paths and IDs remain readable |
| Legacy heuristic output components such as `spatial-mild/order-mild` | `allocation-legacy-mild/task-order-legacy-mild` for newly written legacy-profile examples | Legacy conditions are named honestly, not presented as numeric paper allocations; old paths remain a read fallback |
| Serialized `UEFA`, `uefa-v1`, or versioned LPT identifiers | Historical schema/protocol identities | Kept when required by existing scientific serialization; not public package branding |
| `begin_gcn`, `uefa_gcn`, `fedfst_gat` model identifiers | Existing model registry keys selected by resolved recipes | The synthetic reference model is not interchangeable with the paper GCN or FedFST GAT |
| `GraphKeeper-LP` continual-method key | Paper-facing method label `GraphKeeper` in S7 | The explicit LP implementation key remains in generated low-level commands |
| `fed_pub`, `power_uefa`, and lowercase strategy keys | Paper-facing `FedPUB`, `POWER`, `FedDC`, `FedGTA`, `FedFST`, and `MOTION` | Runner labels resolve to existing strategy keys and validated method configurations; algorithms are not renamed or reimplemented |
| Configuration filenames containing `legacy`, `frozen`, or `promotion` | Resolved configuration paths shown by the plan | Historical filenames are retained for lookup compatibility; their wording is not a new eligibility claim |
| Six-scope `reproduce_50x1.py` utility | `examples/benchmark.py` with S1–S7 | Old six-scope coverage is not a complete paper map; S3 must not gain a learner table |
| Generic method defaults such as Local-only for a supported learner | Scenario-specific main-panel rule, usually FedAvg or the method's native rule | Alternative rules are separate experiments, not aliases of a paper result |
| Source release gates, source manifest, or Git checkout requirement | Normal extracted source installation and execution | No source checksum/commit is required to train or resume; stream integrity checks remain |

For a fresh exact-allocation stream, the layout includes:

```text
<output>/.../uefa-v1/<dataset>/<problem>/<regime>/seed-<n>/
  allocation-alpha-<value>/task-order-<name>/
```

The runner records actual paths; use those instead of inferring them from
old display labels. Low-level generic construction does not become the
exact-allocation paper recipe simply by supplying an alpha. Use
`examples/benchmark.py --stage construct` for the paper construction path.
Changing display names must not silently alter historical stream identity,
tensor content, metrics, eligibility, or checkpoint semantics.

## Known paper-to-artifact discrepancies

S3 is deliberately construction-only. Undefined local ROC-AUC in an earlier
frozen NC-Domain metric contract does not justify inventing learner results
or treating a different complete-metric protocol as the same S3 experiment.

S6 has a substantive unresolved discrepancy: the frozen seed-1 / alpha-0.1
streams in both orders fail validation-query support and are blocked, while
Table 22 reports three-seed results for those anchors. Reproducing those
reported summaries from the blocked inputs would violate the enforced
eligibility policy. The runner must stop, not bypass the audit, omit a seed
without reporting it, or relabel the result as a verified reproduction.

Table 12 lists FedFST under S1/S2, whereas the presented FedFST numerical
extension is synchronized S2 in Tables 14–15 and Table 20 has no S1 FedFST
row. The runner therefore exposes the documented S2 synchronized comparison;
it does not fabricate an S1 FedFST table.

Finally, generic historical configuration defaults can differ from the
manuscript's resolved experiments, including the LP client count and the
aggregation rule attached to a continual learner. The paper runner makes
those experiment choices explicit. This alignment and source cleanup do not
prove that historical numerical payloads were generated by the newly
resolved commands. A numerical reproduction still requires eligible
streams, completed runs, per-seed results, and a disclosed environment.
