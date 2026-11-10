# Corrected and extended LP protocols

The opt-in Facebook constructor `facebook_partition_first_local_base_v2`
assigns disjoint global node ranges to category-local IDs. Its degree features
use only the client's fixed base graph. Saved metadata include the reversible
ID mapping and source CSV checksums. The legacy `S7-*` catalog retains the
original inputs; `LP-Facebook-clean-k3-*` identifies the corrected three-client
inputs. They are different scientific protocols.

`LP-WikiCS-k10-*` and `LP-Collab-k10-*` contain the actual ten-client extension
splits at four anchors, seeds 0–4. Their configs are
`lp_domain_wikics_10client_v1.yaml` and `lp_domain_ogbl-collab_10client_v1.yaml`.
Each uses two stages, 50 communication rounds/stage, one local epoch and five
participants/round. The Wiki-CS graph is repurposed for LP; the Collab split is
GECKO's continual protocol, not the official OGB link-prediction split.

```bash
python tools/run_predefined.py --id LP-WikiCS-k10-s0-a100-synchronized --method Bare --plan
python tools/run_predefined.py --id LP-WikiCS-k10-s0-a100-synchronized --method Bare --device cpu
```

The default GraphKeeper method configuration uses the ridge prediction head.
For a separately labeled dot-product diagnostic, materialize the same split,
use
`configs/gecko_v1/algorithms/graphkeeper_lp_dot_product_diagnostic_v1.yaml`.
Pass that explicit method file with `gecko run --method-config ...`, keeping the stream,
seed, architecture, budget and federated rule unchanged. The method registry
marks this head as diagnostic; it is not an automatic best-method replacement.

The saved payloads fix training positives, evaluation positives, negative
candidate groups and stage-specific message-passing context. Compare methods
on the same split ID; do not regenerate candidates for each learner. Scores
use the declared pessimistic Hits@50 tie rule and positive-query weighting.
