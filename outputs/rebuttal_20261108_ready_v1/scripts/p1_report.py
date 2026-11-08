"""Export signed within-seed HPO effects when complete verified pairs exist."""
import csv
from statistics import mean,stdev
from ready_common import OUT,ROOT,read,sha,values


def export(name,rows):
    if not rows:
        return
    with (OUT/'tables'/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


plan=read(OUT/'provenance/p1_plan.json')['jobs']
metrics={}
for job in plan:
    p=OUT/'runs'/job['id']/'verification.json'
    if p.exists():
        r=read(p)
        assert r['job']==job and r['sha256']==sha(p.parent/'result.json')
        metrics[job['method'],job['variant'],job['seed'],job['alpha'],job['order']]=r['metrics']
    if job['method']=='GEM':
        base=ROOT/job['reusable_baseline']
        assert sha(base)==job['reusable_baseline_sha256']
        order=read(ROOT/job['stream']/'orders.json')['client_orders']
        metrics['GEM','baseline',job['seed'],job['alpha'],job['order']]=values(
            read(base),{int(c):v for c,v in order.items()})
export('p1_cell_metrics.csv',[dict(method=k[0],variant=k[1],seed=k[2],alpha=k[3],order=k[4],**v)
                              for k,v in sorted(metrics.items())])
rows=[]
for method in ('MOTION','GEM'):
    for seed in range(3):
        for metric in ('P','AF','acquisition'):
            effects={}
            for variant in ('baseline','alternative'):
                vals={(a,o):metrics.get((method,variant,seed,a,o)) for a in (.1,100)
                      for o in ('synchronized','hard')}
                if any(v is None for v in vals.values()):
                    continue
                val={k:v[metric] for k,v in vals.items()}
                sign=-1 if metric=='AF' else 1
                effects[variant,'alloc']=100*sign*mean(val[100,o]-val[.1,o] for o in ('synchronized','hard'))
                effects[variant,'order']=100*sign*mean(val[a,'hard']-val[a,'synchronized'] for a in (.1,100))
            if len(effects)!=4:
                continue
            for contrast in ('alloc','order'):
                base,alt=effects['baseline',contrast],effects['alternative',contrast]
                rows.append(dict(method=method,seed=seed,metric=metric,contrast=contrast,
                                 base_pp=base,alternative_pp=alt,delta_pp=alt-base))
export('p1_paired_effects.csv',rows)
summaries=[]
for key in sorted({(r['method'],r['metric'],r['contrast']) for r in rows}):
    a=[r for r in rows if (r['method'],r['metric'],r['contrast'])==key]
    if len(a)==3:
        summaries.append(dict(method=key[0],metric=key[1],contrast=key[2],seeds=3,
                              base_mean_pp=mean(r['base_pp'] for r in a),
                              alt_mean_pp=mean(r['alternative_pp'] for r in a),
                              paired_change_mean_pp=mean(r['delta_pp'] for r in a),
                              paired_change_sd_pp=stdev(r['delta_pp'] for r in a)))
export('p1_paired_summary.csv',summaries)
print('Available cells',len(metrics),'complete paired contrasts',len(rows))
