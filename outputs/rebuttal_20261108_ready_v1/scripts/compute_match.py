"""Reference-only compute diagnostic; calibrate from synchronized GPU time."""
import argparse
import itertools
import math
import time

import torch
from gecko.engine.coordinator import FederatedCoordinator
from gecko.evaluation.reports import json_safe
from tools import e4_gem_solver
from p1 import require_idle_gpu, source_parity
from ready_common import OUT, OLD, read, load, save, sha, short_stream, values

METHODS=('GEM','MOTION')
PROFILE_ROUNDS=20
TOLERANCE=.10


def min_covered_rounds(bundle):
    clients=set(range(bundle.config.partition.num_clients))
    return max(next(k for k in range(1,PROFILE_ROUNDS+1)
                    if set().union(*(set(trace[i]) for i in range(k)))==clients)
               for trace in bundle.participation.trace.values())


def choose_profiles(profiles, lower):
    times={m:{k:sum(sum(stage[:k]) for stage in profiles[m]['seconds_by_stage'])
              for k in range(lower,PROFILE_ROUNDS+1)} for m in METHODS}
    options=[]
    for a,b in itertools.product(times['GEM'],times['MOTION']):
        x,y=times['GEM'][a],times['MOTION'][b]
        mismatch=abs(x-y)/max(x,y)
        if mismatch<=TOLERANCE:
            options.append((min(x,y),-mismatch,a,b))
    if not options:
        raise RuntimeError('No planned pair fits the synchronized training-time tolerance and participant coverage.')
    _,_,a,b=max(options)
    return dict(rounds={'GEM':a,'MOTION':b},
                projected_seconds={'GEM':times['GEM'][a],'MOTION':times['MOTION'][b]},
                tolerance=TOLERANCE, selection_used_test=False)


def execute(method, path, bundle, rounds, phase):
    gpu_lock=require_idle_gpu()
    source_parity()
    cfgpath=OLD/'configs'/f'HPO-{method}-original.json'
    if method=='GEM':
        e4_gem_solver.install()
    validation=phase=='profile'
    b=short_stream(bundle,rounds,f'compute-{phase}-{method}',validation_only=validation)
    digest={'phase':phase,'method':method,'rounds':rounds,'gpu':torch.cuda.get_device_name(0),
            'torch':torch.__version__,'stream_sha256':sha(path/'manifest.json'),
            'config_sha256':sha(cfgpath),'solver_sha256':sha(OUT.parents[1]/'tools/e4_gem_solver.py')
            if method=='GEM' else None}
    dest=OUT/'runs'/f'compute-{phase}-{method}.json'
    if dest.exists():
        prior=read(dest)
        assert prior['binding']==digest
        return prior
    co=FederatedCoordinator(b,'fedavg' if method=='GEM' else 'motion',
                            'GEM' if method=='GEM' else 'Bare',
                            model_name='begin_gcn',model_seed=0,device='cuda:0',
                            method_config=read(cfgpath))
    runtime=co._stateful_runtime
    original=runtime._run_round
    elapsed=[[] for _ in range(8)]
    def timed_round(stage,index):
        torch.cuda.synchronize()
        start=time.perf_counter()
        record=original(stage,index)
        torch.cuda.synchronize()
        elapsed[stage].append(time.perf_counter()-start)
        return record
    runtime._run_round=timed_round
    result=json_safe(co.run())
    assert len(result['rounds'])==rounds*8 and all(
        r['participants']==5 and r['empty_updates']==0 and math.isfinite(r['mean_training_loss'])
        for r in result['rounds'])
    row={'binding':digest,'seconds_by_stage':elapsed,'seconds_total':sum(map(sum,elapsed)),
         'optimizer_steps':sum(r['participants'] for r in result['rounds'])*
             result['training_budget']['effective_optimizer_steps_per_round'],
         'trainable_parameters':result['trainable_parameter_count'],
         'peak_cuda_bytes':result['peak_memory_bytes'],'strategy_diagnostics':result['strategy_diagnostics']}
    if not validation:
        row['test_metrics']=values(result,b.orders.client_orders)
        row['evaluation_seconds']=sum(r['evaluation_runtime_seconds'] for r in result['stages'])
    save(dest,row)
    return row


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=('dry-run','profile','compare'))
    args=ap.parse_args()
    torch.set_num_threads(1)
    path,b=load('nc_class',0,100,'synchronized')
    lower=min_covered_rounds(b)
    if args.mode=='dry-run':
        save(OUT/'provenance/compute_dry_run.json',{'seed':0,'alpha':100,'order':'synchronized',
             'profile_rounds_per_stage':PROFILE_ROUNDS,'minimum_rounds_all_clients':lower,
             'time_tolerance':TOLERANCE,'manifest_sha256':sha(path/'manifest.json')})
        print('Compute dry run passed; all clients appear per stage by round',lower)
        return
    if args.mode=='profile':
        rows={m:execute(m,path,b,PROFILE_ROUNDS,'profile') for m in METHODS}
        a,c=(rows[m]['binding'] for m in METHODS)
        assert all(a[k]==c[k] for k in ('gpu','torch','stream_sha256'))
        plan=choose_profiles(rows,lower)
        plan.update(profile_sha256={m:sha(OUT/'runs'/f'compute-profile-{m}.json') for m in METHODS})
        save(OUT/'provenance/compute_matched_plan.json',plan)
        return
    plan=read(OUT/'provenance/compute_matched_plan.json')
    assert all(plan['profile_sha256'][m]==sha(OUT/'runs'/f'compute-profile-{m}.json') for m in METHODS)
    rows={m:execute(m,path,b,plan['rounds'][m],'compare') for m in METHODS}
    for m in METHODS:
        baseline=read(OUT/'runs'/f'compute-profile-{m}.json')['binding']
        assert all(rows[m]['binding'][k]==baseline[k] for k in ('gpu','torch','stream_sha256','config_sha256'))
    a,c=(rows[m]['seconds_total'] for m in METHODS)
    mismatch=abs(a-c)/max(a,c)
    save(OUT/'provenance/compute_matched_result.json',
         {'status':'MATCHED' if mismatch<=TOLERANCE else 'BUDGET_MISMATCH',
          'training_time_mismatch':mismatch,'rows':rows})


if __name__=='__main__':
    main()
