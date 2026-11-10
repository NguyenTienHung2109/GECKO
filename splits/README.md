# Predefined splits

`predefined/catalog.json` enumerates **228 fixed stream cells**. A cell is one
seed × allocation × order; it is reused by all compared methods. It is not a
training run. The catalog records client counts, stages, round budgets and the
original stream's eligibility. No accuracy values or checkpoint weights are
included.

| Family | Seeds | Alpha | Orders | Clients | Cells |
| --- | --- | --- | --- | --- | --- |
| Original S1: Arxiv Task-IL | 0–2 | 0.1, 100 | Sync, Hard | 10 | 12 |
| Original S2: Arxiv Class-IL | 0–2 | 0.1, 1, 10, 100 | Sync, Hard | 10 | 24 |
| Original S3: Proteins Domain-IL | 0–2 | 0.1, 1, 10, 100 | Sync, Hard | 10 | 24 |
| Original S4: Bitcoin Task-IL | 0–2 | 0.1, 100 | Sync, Hard | 10 | 12 |
| Original S5: Bitcoin Class-IL | 0–2 | 0.1, 1, 10, 100 | Sync, Hard | 10 | 24 |
| Original S6: Bitcoin Domain-IL | 0–2 | 0.1, 100 | Sync, Hard | 10 | 12 |
| Original S7: Facebook LP | 0–4 | 0.1, 1, 10, 100 | Sync, binary mismatch | 3 | 40 |
| Facebook clean LP | 0–4 | 0.1, 1, 10, 100 | Sync, binary mismatch | 3 | 40 |
| Wiki-CS LP extension | 0–4 | 0.1, 100 | Sync, binary mismatch | 10 | 20 |
| OGBL-Collab LP extension | 0–4 | 0.1, 100 | Sync, binary mismatch | 10 | 20 |

`hard` and `binary_mismatch` are the serialized names for the corresponding
Unsync task-order conditions. Federated communication remains synchronous.
Base/Alloc./Order/Joint are (100, Sync)/(0.1, Sync)/(100, Unsync)/(0.1, Unsync).
S1–S3 use 8 stages, S4–S5 use 3, S6 uses 4, LP uses 2; all have 50 rounds/stage
and one local epoch. Exact participation traces, not just fractions, are saved.
Original S3 is construction-only. The two ineligible S6 cells remain included
for inspection and are not silently removed. Original S7 and Facebook clean
have three clients; the two new LP datasets have ten.

## Content and provenance

Each compressed cell manifest includes:

- resolved scientific configuration;
- immutable global task IDs, class/domain assignments and masks;
- client ownership, local/global coordinate mappings and boundary masks;
- exact train/validation/test query IDs, labels and local coordinates;
- fixed LP positive/negative candidates and stage-specific context;
- canonical and client-specific task orders and inverse mappings;
- complete per-stage/per-round participation;
- original stream ID, scientific fingerprint and source artifact checksums.

JSON mapping tags preserve integer keys and tuples. Tensors are compressed
NumPy arrays (`allow_pickle=False`), with dtype/shape/content SHA-256 checks.
Tiny arrays are inline; large integer-to-tensor dictionaries use packed arrays.
Deduplication changes storage only. Host paths are removed; original source
container checksums are provenance, not claims that reserialization is byte
identical. This format never loads a Python pickle.

For NC, the raw graph, features and Proteins edge attributes come from the
public loader. Their hashes must match. Saved ownership/mappings reconstruct
induced local graph tensors, again checked against the original tensor hashes.
A mismatch stops import; it never falls back to fresh partitions. LC/LP's
compact derived tensors are included under the upstream dataset terms.

## Inspect or run

```bash
python -m gecko.data.predefined verify
python -m gecko.data.predefined materialize --id S5-s0-a100-synchronized
python tools/run_predefined.py --id S5-s0-a100-synchronized --method GEM --plan
```

Programmatic access does not require materializing a new on-disk stream:

```python
from pathlib import Path
from gecko.data.predefined import load_predefined
bundle = load_predefined(Path('splits/predefined'),
                         'S5-s0-a100-synchronized', Path('data'))
owner = bundle.partition.node_owner
train_ids = bundle.scenario.query_ids_by_task_split[0]['train']
validation_ids = bundle.evaluation_shards[0][0].validation_query_ids
client_order = bundle.orders.client_orders[0]
participants = bundle.participation.trace[0][0]
```

The importer runs no assignment/order/participation sampler. The normal training
runner recomputes audits and enforces eligibility. Imported containers use schema
2 and new container fingerprints; all fixed scientific tensors are checked.
The source catalog covers the original paper's primary grids plus the explicit
LP extensions above. It is not an archive of every later seed-extension or
exploratory diagnostic campaign. Full source construction APIs remain available
for additional seeds and settings.
