#!/usr/bin/env python3
"""Stability and ablation checks computed from existing scan rows: bootstrap enlargement
and multiple seeds, per-admission heterogeneity, leave-one-patient-out influence,
coverage-tertile stratification, and persistence-only and gap-only ablations.
"""
from alarmreplay.paths import WORK as R, COVERAGE
import csv,glob,json,os,sys,collections
SEL={"ecg-numLimit-heartRate-high":(8,16),"ecg-numLimit-heartRate-low":(0,30),
     "ecgResp-numLimit-respRate-low":(16,8),"spO2-numLimit-satO2-low":(16,16)}
NAI=(0,30)
COVCH={"ecg-numLimit-heartRate-high":"h_min","ecg-numLimit-heartRate-low":"h_min",
       "ecgResp-numLimit-respRate-low":"r_min","spO2-numLimit-satO2-low":"s_min"}
stage=sys.argv[1]
CACHE=f"{R}/sensitivity/_tier1_cache.json"
if stage=="1":
    cov={}
    for row in csv.DictReader(open(COVERAGE)):
        try: cov[row["aCSN"]]={"h_min":float(row["h_min"] or 0),"r_min":float(row["r_min"] or 0),"s_min":float(row["s_min"] or 0)}
        except Exception: pass
    # per (channel, cell, split): per-patient sums; per-admission naive ratios; coverage join
    agg=collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0]*6))  # key (split,ch,tau,g) -> aMRN -> sums
    adm_naive=collections.defaultdict(list)  # TEST per-admission naive (recon,rec,cov)
    grid=collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0]*6))  # FIT pooled per cell
    seen=set()
    for split in ("FIT","TUNE","TEST"):
        for fp in sorted(glob.glob(f"{R}/delay_model/a2_{split}_rows_*.csv")):
            for r in csv.DictReader(open(fp)):
                key=(split,r["aMRN"],r["aCSN"],r["channel"],r["tau"],r["g"])
                if key in seen: continue
                seen.add(key)
                ch=r["channel"]; tau=int(r["tau"]); g=int(r["g"])
                vals=[float(r["rec_runs"]),float(r["match"]),float(r["recon_runs"]),float(r["recon_secs"]),float(r["rec_secs"]),float(r["onset_within10"])]
                if split=="FIT":
                    gg=grid[(ch,tau,g)]["_pooled"]
                    for i in range(6): gg[i]+=vals[i]
                keep=((tau,g)==SEL[ch]) or ((tau,g)==NAI)
                if keep:
                    a=agg[(split,ch,tau,g)][r["aMRN"]]
                    for i in range(6): a[i]+=vals[i]
                if split=="TEST" and (tau,g)==NAI and vals[4]>0:
                    c=cov.get(r["aCSN"],{}).get(COVCH[ch],0.0)
                    adm_naive[ch].append([vals[3],vals[4],c])
    out={"agg":{ "|".join(map(str,k)):v for k,v in ((k,dict(v)) for k,v in agg.items())},
         "adm_naive":adm_naive,
         "grid":{ "|".join(map(str,k)):v["_pooled"] for k,v in grid.items()}}
    json.dump(out,open(CACHE,"w"))
    print("stage1 done: cells",len(agg),"grid cells",len(out["grid"]),"adm rows",{k:len(v) for k,v in adm_naive.items()})
else:
    import numpy as np
    d=json.load(open(CACHE))
    res={"bootstrap":{},"heterogeneity":{},"influence":{},"coverage_tertiles":{},"ablation":{}}
    # ---- bootstrap with 3 seeds, B=2000
    for key,per in d["agg"].items():
        split,ch,tau,g=key.split("|")
        if split!="TEST": continue
        pats=sorted(per); X=np.array([per[p][3] for p in pats]); Y=np.array([per[p][4] for p in pats])
        m=Y>0; X,Y=X[m],Y[m]; n=len(X); rho=X.sum()/Y.sum()
        cis={}
        for seed in (20260831,7,12345):
            rng=np.random.default_rng(seed)
            idx=rng.integers(0,n,size=(2000,n))
            rr=X[idx].sum(axis=1)/Y[idx].sum(axis=1)
            cis[str(seed)]=[round(float(np.percentile(rr,2.5)),4),round(float(np.percentile(rr,97.5)),4)]
        los=[c[0] for c in cis.values()]; his=[c[1] for c in cis.values()]
        # leave-one-patient-out influence
        tx,ty=X.sum(),Y.sum()
        loo=(tx-X)/(ty-Y); infl=float(np.max(np.abs(loo-rho)))
        lab=f"{ch}|tau{tau}g{g}"
        res["bootstrap"][lab]={"n_patients":int(n),"rho":round(float(rho),4),"ci_by_seed":cis,
            "max_endpoint_spread_across_seeds":round(max(max(his)-min(his),max(los)-min(los)),4)}
        res["influence"][lab]={"max_abs_change_drop_one_patient":round(infl,4),
                                "relative":round(infl/rho,4)}
    # ---- per-admission heterogeneity + coverage tertiles (naive, TEST)
    for ch,rows in d["adm_naive"].items():
        a=np.array(rows,dtype=float); r=a[:,0]/a[:,1]
        res["heterogeneity"][ch]={"n_admissions":int(len(r)),"median":round(float(np.median(r)),3),
            "q25":round(float(np.percentile(r,25)),3),"q75":round(float(np.percentile(r,75)),3),
            "share_above_1":round(float((r>1).mean()),3)}
        c=a[:,2]; t=np.quantile(c,[1/3,2/3])
        tert={}
        for j,(lo,hi) in enumerate([(-1,t[0]),(t[0],t[1]),(t[1],1e12)]):
            m=(c>lo)&(c<=hi)
            tert[f"T{j+1}"]={"n":int(m.sum()),"pooled_ratio":round(float(a[m,0].sum()/a[m,1].sum()),3),
                             "median_cov_min":round(float(np.median(c[m])),0)}
        res["coverage_tertiles"][ch]=tert
    # ---- ablation: best tau-only (g=30) and best g-only (tau=0) by FIT pooled F1; report TEST metrics
    grid={}
    for k,v in d["grid"].items():
        ch,tau,g=k.split("|"); grid[(ch,int(tau),int(g))]=v
    def f1(v):
        rec=v[1]/v[0] if v[0] else 0; pre=v[1]/v[2] if v[2] else 0
        return 2*pre*rec/(pre+rec) if pre+rec else 0
    for ch in SEL:
        best_t=max(((t,30) for t in (0,4,8,12,16,24)),key=lambda c: f1(grid[(ch,c[0],c[1])]))
        best_g=max(((0,g) for g in (8,16,30)),key=lambda c: f1(grid[(ch,c[0],c[1])]))
        def met(c):
            v=grid[(ch,c[0],c[1])]
            rec=v[1]/v[0] if v[0] else 0; pre=v[1]/v[2] if v[2] else 0
            return {"recall":round(rec,3),"precision":round(pre,3),"f1":round(f1(v),3),
                    "burden_secs_ratio":round(v[3]/v[4],3) if v[4] else None,
                    "onset10_among_matched":round(v[5]/v[1],3) if v[1] else None}
        res["ablation"][ch]={"full_selected":{"cell":list(SEL[ch]),"fit":met(SEL[ch])},
                             "tau_only_best":{"cell":list(best_t),"fit":met(best_t)},
                             "g_only_best":{"cell":list(best_g),"fit":met(best_g)},
                             "note":"ablation confined to FIT grid by the one-look TEST policy"}
    json.dump(res,open(f"{R}/sensitivity/STABILITY_AND_ABLATION.json","w"),indent=1)
    for k,v in res["bootstrap"].items(): print(k,v["rho"],"spread",v["max_endpoint_spread_across_seeds"],"| infl",res["influence"][k]["relative"])
    for ch,v in res["heterogeneity"].items(): print("het",ch,v)
    print("tertiles:",json.dumps(res["coverage_tertiles"])[:400])
    print("ablation cells:",{k:(v["tau_only_best"],v["g_only_best"]) for k,v in res["ablation"].items()})
