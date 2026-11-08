"""Equal-width, equal-parameter LC decoder ablation on the original owned graph."""
import argparse
import itertools
import math
from pathlib import Path

import torch
from gecko.engine.coordinator import FederatedCoordinator
from gecko.evaluation.metrics import macro_f1
from gecko.evaluation.reports import json_safe
from gecko.models.backbones import LegacyGCNAdapter
from p1 import require_idle_gpu
from ready_common import OUT, load, short_stream, save, sha, read, values


class PairDecoderGCN(LegacyGCNAdapter):
    def __init__(self, variant, **kwargs):
        super().__init__(**kwargs)
        self.variant=variant

    def forward_queries(self, features, edges, queries, problem):
        assert problem.upper()=='LC'
        h=self._torch_encode_nodes(features,edges)
        src,dst=h[queries[:,0]],h[queries[:,1]]
        pair=src*dst
        if self.variant=='directed':
            pair=pair+(src-dst)
        pair=self.original.dropout(pair)
        for layer in self.original.linears:
            pair=self.original.dropout(self.original.activation(layer(pair)))
        return self.original.classifier(pair)


def run(seed,alpha,variant,smoke):
    gpu_lock=None if smoke else require_idle_gpu()
    path,b=load('lc_class',seed,alpha,'hard')
    manifest_sha=sha(path/'manifest.json')
    if smoke:
        b=short_stream(b,2,'lc-validation-smoke',validation_only=True)
    rounds=b.config.training.rounds_per_stage
    assert rounds==(2 if smoke else 50) and b.scenario.num_tasks==3
    dest=OUT/'runs'/f"lc-{'smoke' if smoke else 'full'}-s{seed}-a{alpha:g}-{variant}"
    result_path=dest/'result.json'
    binding={'seed':seed,'alpha':alpha,'order':'hard','variant':variant,'rounds':rounds,
             'device':'cpu' if smoke else 'cuda:0','input_manifest_sha256':manifest_sha,
             'script_sha256':sha(Path(__file__)),'evaluation':'validation only' if smoke else 'original'}
    if result_path.exists():
        result=read(result_path)
        assert result['binding']==binding
        return result
    cfg={'schema':'uefa-method-config','version':2,'name':'legacy_adapter_v1',
         'strategy':{'name':'fedavg','parameters':{}},'continual_method':{'name':'Bare','parameters':{}}}
    co=FederatedCoordinator(b,'fedavg','Bare',model_name='begin_gcn',model_seed=seed,
        device=binding['device'],method_config=cfg,
        model_factory=lambda:PairDecoderGCN(variant,input_size=b.scenario.num_features,
            output_size=b.scenario.num_classes,hidden_size=b.config.training.hidden_size,
            num_layers=b.config.training.num_layers,problem_type='LC',incremental_type='class'))
    result=json_safe(co.run())
    assert len(result['rounds'])==rounds*3
    assert all(r['empty_updates']==0 and r['participants']==5 and math.isfinite(r['mean_training_loss'])
               for r in result['rounds'])
    assert result['in_memory_stream_tensors_unchanged']
    fit=[]
    with torch.no_grad():
        for c,client in co.clients.items():
            client.load_shared_state(co.global_state)
            client.model.eval()
            for t,shard in b.shards[c].items():
                ev=b.evaluation_shards[c][t]
                splits=[('train',shard.train_queries,shard.train_labels),
                        ('val',ev.validation_queries,b.scenario.labels[ev.validation_query_ids])]
                if not smoke:
                    splits.append(('test',ev.test_queries,b.scenario.labels[ev.test_query_ids]))
                for name,queries,labels in splits:
                    logits=client._forward(client.model,queries,shard.context_edge_index).cpu()
                    assert logits.isfinite().all()
                    pred=logits.argmax(-1)
                    fit.append(dict(client=c,task=t,split=name,n=len(labels),
                        accuracy=float((pred==labels).float().mean()),macro_f1=macro_f1(logits,labels),
                        histogram=torch.bincount(pred,minlength=b.scenario.num_classes).tolist()))
    record={'binding':binding,'fit':fit,'metrics':values(result,b.orders.client_orders),
            'result':result}
    save(result_path,record)
    print(variant,record['metrics'],len(fit),flush=True)
    return record


def main():
    torch.set_num_threads(1)
    ap=argparse.ArgumentParser()
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--seed',type=int,choices=range(3),default=0)
    ap.add_argument('--alpha',type=float,choices=(.1,100),default=100)
    args=ap.parse_args()
    results=[run(args.seed,args.alpha,v,args.smoke) for v in ('product','directed')]
    a,b=(r['result'] for r in results)
    assert a['initial_model_state_digest']==b['initial_model_state_digest']
    assert a['trainable_parameter_count']==b['trainable_parameter_count']
    assert [r['participant_ids'] for r in a['rounds']]==[r['participant_ids'] for r in b['rounds']]
    print('Matched initialization, parameters and participation:',a['trainable_parameter_count'])
    if args.smoke:
        save(OUT/'provenance/lc_smoke_pass.json',{'scope':'CPU validation-only, 2 rounds per stage',
             'parameters':a['trainable_parameter_count'],'seed':args.seed,'alpha':args.alpha})


if __name__=='__main__':
    main()
