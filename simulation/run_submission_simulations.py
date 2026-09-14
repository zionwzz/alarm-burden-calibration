#!/usr/bin/env python3
"""Reproduce all submitted simulations and saved patient-bootstrap intervals.

This full run is computationally intensive. It uses four million pseudo-patients
per scenario for truth, 500 primary replicates and 400 size-check replicates.
"""
from pathlib import Path
import os, sys, time, json, concurrent.futures as cf
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'simulation'))
import simulate_linked_reward as s
OUT=R/'simulation/results/_jobs';OUT.mkdir(parents=True,exist_ok=True)
def work(job):
    kind,sc,n,reps=job; start=time.time();rows=[]
    ans=s.run_scenario(sc,reps,n,200,s.SEED,pre_arm=True,record=rows)
    ans['_label']=s.SCENARIO_LABEL[sc]
    ans['_patient_effect_normalization']='Known population mean; independent patient clusters'
    stem=f'{kind}_{sc}_{n}'
    s.write_json(ans,OUT/(stem+'.json'));s.write_replicates(rows,OUT/(stem+'.csv'))
    return (job,time.time()-start)
if __name__=='__main__':
    jobs=[('zero','J',n,400) for n in [600,1200]]+[('sweep','A',n,400) for n in [2400,1200,600]]+ [('main',sc,s.SCENARIO_PATIENTS.get(sc,300),500) for sc in s.ALL_SCENARIOS.split(',')]
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    if args.workers<1: parser.error('--workers must be positive')
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for ans in ex.map(work,jobs): print('FINISHED',ans,flush=True)
    # Merge all scenario outputs. Scenario A at n=300 supplies its first 400 reps for the sweep.
    import csv, numpy as np
    results={};rows=[]
    for sc in s.ALL_SCENARIOS.split(','):
        stem=f'main_{sc}_{s.SCENARIO_PATIENTS.get(sc,300)}'
        results[sc]=json.loads((OUT/(stem+'.json')).read_text())
        with open(OUT/(stem+'.csv')) as f: rows+=list(csv.DictReader(f))
    dest=R/'simulation/results';s.write_json(results,dest/'SIMULATION_LINKED_REWARD.json')
    with open(dest/'SIMULATION_LINKED_REWARD_REPLICATES.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=s.REPLICATE_FIELDS);w.writeheader();w.writerows(rows)
    # Independent n=300 sweep run is cheap enough to preserve exact run_sweep's summary contract.
    rr=[];base=s.run_scenario('A',400,300,200,s.SEED,pre_arm=True,record=rr)
    s.write_json(base,OUT/'sweep_A_300.json');s.write_replicates(rr,OUT/'sweep_A_300.csv')
    sweep={'_scenario':'H','_label':s.SCENARIO_LABEL['H'],'_dgp':'A','_settings':dict(sizes=[300,600,1200,2400],n_reps=400,n_boot=200,seed=s.SEED,prespecified_map_arm=True,truth_patients=s.TRUTH_PATIENTS,truth_seed=s.TRUTH_SEED['A'],patient_effect_normalization='Known population mean'),'sizes':[]}
    allrows=[]
    for n in [300,600,1200,2400]:
        x=json.loads((OUT/f'sweep_A_{n}.json').read_text())
        row=dict(n_patients=n,truth=x['_truth'],truth_mcse=x['_truth_mcse'])
        for key in ['lrc_d','lrc_dz','lrc_dz_pre']:row[key]=x[key]
        row['_diagnostics']=x['_diagnostics'];sweep['sizes'].append(row)
        with open(OUT/f'sweep_A_{n}.csv') as f:
            for q in csv.DictReader(f):q['scenario']='H';allrows.append(q)
    sweep['_finding']=s._sweep_finding(sweep);s.write_json(sweep,dest/'SIMULATION_SAMPLE_SIZE.json')
    with open(dest/'SIMULATION_SAMPLE_SIZE_REPLICATES.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=s.REPLICATE_FIELDS);w.writeheader();w.writerows(allrows)
    for n in [600,1200]:
        import shutil
        shutil.copy2(OUT/f'zero_J_{n}.json',dest/f'ZERO_CHECK_{n}.json')
        shutil.copy2(OUT/f'zero_J_{n}.csv',dest/f'ZERO_CHECK_{n}_REPLICATES.csv')
    print('ALL SIMULATIONS COMPLETE',sweep['_finding'],flush=True)
