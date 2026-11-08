from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import torch

from gecko.data.streams import load_trusted_legacy_stream
from gecko.evaluation.reports import json_safe

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
OLD = ROOT/'outputs/rebuttal_20261107_remaining_v1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(json_safe(obj),indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def get_stream(scenario, seed, alpha, order):
    key=f'{scenario}:{seed}:{alpha:g}:{order}'
    return ROOT/read(OUT/'provenance/streams.json')[key]


def load(scenario, seed, alpha, order):
    path=get_stream(scenario,seed,alpha,order)
    return path,load_trusted_legacy_stream(path,verify=True)


def short_stream(b, rounds, label, validation_only=False):
    cfg=replace(b.config,training=replace(b.config.training,rounds_per_stage=rounds))
    trace={s:{r:clients for r,clients in rr.items() if r<rounds}
           for s,rr in b.participation.trace.items()}
    part=replace(b.participation,trace=trace,rounds_per_stage=rounds)
    ev=b.evaluation_shards
    if validation_only:
        ev={c:{t:replace(sh,test_queries=sh.validation_queries,
                          test_query_ids=sh.validation_query_ids) for t,sh in ts.items()}
            for c,ts in ev.items()}
    digest=hashlib.sha256(f'{b.stream_hash}:{rounds}:{label}:{validation_only}'.encode()).hexdigest()
    return replace(b,config=cfg,participation=part,evaluation_shards=ev,
                   stream_id=b.stream_id+'-'+label,stream_hash=digest)


def values(result, orders):
    from statistics import mean
    acq, final, af = [], [], []
    for c, stages in enumerate(result['A[k,s,t]']):
        for t in range(len(stages)):
            history=[s[t] for s in stages[orders[c].index(t):] if s[t] is not None]
            if history:
                acq.append(history[0]);final.append(history[-1]);af.append(max(history)-history[-1])
    return {'P':mean(final),'acquisition':mean(acq),'AF':mean(af),'cells':len(final)}
