"""Prepare/run matched NC sensitivity comparisons. Full jobs require an idle GPU."""
import argparse
import fcntl
import itertools
import math
import os
import subprocess
import sys
import tarfile
from dataclasses import replace
from pathlib import Path

import dgl
import torch
from gecko.algorithms.method_config import validate_method_config
from gecko.engine.coordinator import FederatedCoordinator
from gecko.evaluation.reports import json_safe
from tools import e4_gem_solver
from ready_common import OUT, ROOT, OLD, get_stream, load, read, save, sha, values

E4=ROOT/'outputs/e4_20260929'
Q4=ROOT/'outputs/five_seed_extension_20261003/e4'


def source_parity():
    with tarfile.open(OUT/'provenance/source_before.tar.gz') as archive:
        names={m.name for m in archive if m.isfile() and m.name.startswith('src/gecko/') and m.name.endswith('.py')}
        assert names=={str(p.relative_to(ROOT)) for p in (ROOT/'src/gecko').rglob('*.py')}
        for name in sorted(names):
            with archive.extractfile(name) as f:
                assert f.read()==(ROOT/name).read_bytes(), name
    assert len(names)>200
    return len(names)


def prepare():
    source_count=source_parity()
    checks={r['key']:r for r in read(Q4/'verification.json')['records']}
    assert read(Q4/'verification.json')['passed']
    plan=[]
    for seed, alpha, order in itertools.product(range(3),(.1,100),('synchronized','hard')):
        stream=get_stream('nc_class',seed,alpha,order)
        manifest=stream/'manifest.json'
        profile='sync' if order=='synchronized' else 'hard0'
        key=f'a{alpha:g}-s{seed}-{profile}-GEM'
        baseline=E4/'runs'/key/'result.json'
        assert checks[key]['passed'] and checks[key]['checkpoint_verified']
        assert checks[key]['result_sha256']==sha(baseline)
        old=read(baseline)
        assert old['e4']['source_manifest_sha256']==sha(manifest)
        assert old['e4']['solver']['source_sha256']==sha(ROOT/'tools/e4_gem_solver.py')
        assert old['method_config']==read(OLD/'configs/HPO-GEM-original.json')
        assert old['model_seed']==seed and old['resolved_model']=='begin_gcn'
        assert len(old['rounds'])==400 and all(r['participants']==5 for r in old['rounds'])
        assert read(E4/'orders'/f's{seed}-{profile}.json')['orders']['client_orders']==read(stream/'orders.json')['client_orders']
        for method,variant in [('GEM','alternative'),('MOTION','baseline'),('MOTION','alternative')]:
            cfg = OUT/'configs/GEM-alternative.json' if method=='GEM' else OLD/'configs'/(
                'HPO-MOTION-original.json' if variant=='baseline' else 'HPO-MOTION-alternative.json')
            plan.append(dict(id=f'p1-{method}-{variant}-s{seed}-a{alpha:g}-{order}',seed=seed,alpha=alpha,
                             order=order,method=method,variant=variant,
                             stream=str(stream.relative_to(ROOT)),manifest_sha256=sha(manifest),
                             config=str(cfg.relative_to(ROOT)),config_sha256=sha(cfg) if cfg.exists() else None,
                             reusable_baseline=str(baseline.relative_to(ROOT)) if method=='GEM' else None,
                             reusable_baseline_sha256=sha(baseline) if method=='GEM' else None))
    assert len(plan)==36
    for entry in plan:
        cfg=ROOT/entry['config']
        if cfg.is_file():
            validate_method_config(read(cfg),expected_strategy='fedavg' if entry['method']=='GEM' else 'motion',
                expected_continual_method='GEM' if entry['method']=='GEM' else 'Bare',
                problem_type='NC',incremental_setting='class')
    # Read one stream per condition through the real loader, then check full
    # participation traces and order pairing on the original manifests.
    seen={}
    for job in plan[::3]:
        path,b=load('nc_class',job['seed'],job['alpha'],job['order'])
        assert b.config.training.rounds_per_stage==50 and b.config.training.local_epochs_per_round==1
        assert b.scenario.num_tasks==8 and b.config.partition.num_clients==10
        assert len(b.participation.trace)==8
        assert all(len(r)==50 and all(len(c)==5 for c in r.values()) for r in b.participation.trace.values())
        manifest=read(path/'manifest.json')['artifact_checksums']
        key=(job['seed'],job['alpha'])
        paired={k:manifest[k] for k in ('scenario.pt','evaluation.pt','client_query_shards.pt',
                                       'ownership.pt','client_subgraph_index.pt','participation.json')}
        assert seen.setdefault(key,paired)==paired
        del b
    save(OUT/'provenance/p1_plan.json',{'jobs':plan,'e4_source_files_matched':source_count,
                                      'gem_baselines_verified':12,'scope':'matched S2 sensitivity diagnostic'})
    print('P1 dry run passed:',len(plan),'new jobs; 12 GEM baselines verified; source files',source_count)


def require_idle_gpu():
    for session in ('rebuttal-s3-ref-v1','rebuttal-s3-expand-v1'):
        if subprocess.run(['tmux','has-session','-t','='+session],capture_output=True).returncode==0:
            raise SystemExit('S3 owns this host GPU. Use a separate compatible idle GPU or wait.')
    command=['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']
    gpu=os.environ.get('GECKO_RESERVED_GPU_UUID')
    if gpu:
        if not gpu.startswith('GPU-') or os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu:
            raise SystemExit('The reserved GPU must be the only CUDA-visible device.')
        command.extend(['--id',gpu])
    processes=subprocess.check_output(command,text=True)
    if processes.strip():
        raise SystemExit('A CUDA compute process is active. Training was not started.')
    lock=(OLD/'artifacts/s3_gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return lock


def run(job):
    lock=require_idle_gpu()
    method=job['method']
    cfg=ROOT/job['config']
    assert job['config_sha256'] and job['config_sha256']==sha(cfg)
    env=read(E4/'provenance/environment.json')
    assert str(torch.__version__)==env['torch'] and dgl.__version__==env['dgl']
    path,b=load('nc_class',job['seed'],job['alpha'],job['order'])
    assert sha(path/'manifest.json')==job['manifest_sha256']
    target=OUT/'runs'/job['id']
    receipt=target/'verification.json'
    if receipt.exists():
        assert read(receipt)['sha256']==sha(target/'result.json')
        return
    binding={'job':job,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
             'runner_sha256':sha(Path(__file__)),'source_files':source_parity()}
    if method=='GEM':
        original=ROOT/job['reusable_baseline']
        assert sha(original)==job['reusable_baseline_sha256']
        baseline=read(original)
        binding.update(solver=e4_gem_solver.SOLVER_VERSION,solver_sha256=sha(ROOT/'tools/e4_gem_solver.py'))
        e4_gem_solver.install()
    bind=target/'binding.json'
    if bind.exists():
        assert read(bind)==binding
    else:
        save(bind,binding)
    checks=sorted((target/'checkpoints').glob('stage-*.manifest.json'))
    class Logger:
        def log(self, record, step=None):
            if 'stage_runtime_seconds' in record:
                print(job['id'],record['stage'],record['stage_runtime_seconds'],flush=True)
    co=FederatedCoordinator(b,'fedavg' if method=='GEM' else 'motion',
                            'GEM' if method=='GEM' else 'Bare',
                            model_name='begin_gcn',model_seed=job['seed'],device='cuda:0',
                            method_config=read(cfg),source_sha=sha(bind),run_logger=Logger(),
                            checkpoint_dir=target/'checkpoints',resume_from=checks[-1] if checks else None)
    r=json_safe(co.run())
    assert len(r['rounds'])==400 and all(x['participants']==5 and x['empty_updates']==0 and
                 math.isfinite(x['mean_training_loss']) for x in r['rounds'])
    assert r['in_memory_stream_tensors_unchanged'] and r['primary_metric_support']['complete']
    if method=='GEM':
        assert r['initial_model_state_digest']==baseline['initial_model_state_digest']
        assert [x['participant_ids'] for x in r['rounds']]==[x['participant_ids'] for x in baseline['rounds']]
        r['bvls_counters_current_process']=dict(e4_gem_solver.COUNTERS)
    result=values(r,b.orders.client_orders)
    assert abs(result['P']-r['summary']['final_average_performance'])<1e-6
    assert abs(result['AF']-r['summary']['average_forgetting'])<1e-6
    save(target/'result.json',r)
    save(receipt,{'job':job,'sha256':sha(target/'result.json'),'metrics':result})
    print('VERIFIED',job['id'],result,flush=True)


def main():
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--job')
    args=parser.parse_args()
    if args.prepare:
        prepare()
        return
    plan=read(OUT/'provenance/p1_plan.json')['jobs']
    run(next(x for x in plan if x['id']==args.job))

if __name__=='__main__':
    main()
