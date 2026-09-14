#!/usr/bin/env python3
"""Recorded burden, silencing and burden ratios by diagnosis-derived oncology tier and by
cancer group. Descriptive only: no selection and no criteria.
"""
import csv,glob,json,os,collections
import numpy as np
from alarmreplay.paths import WORK as R, COHORT as SAMP
O=f"{R}/sensitivity"
tier={r["aCSN"]:r["tier"] for r in csv.DictReader(open(SAMP))}
site={r["aCSN"]:r["cancer_group"] for r in csv.DictReader(open(f"{R}/cache/cancer_group_by_admission.csv"))}  # the malignant-code rule of analysis/cancer_groups.py, strict cohort
splitmap={r["aMRN"]:r["split"] for r in csv.DictReader(open(f"{R}/design/SPLIT_ASSIGNMENT.csv"))}
TIERS=["strict_onc","onc_context_nonstrict","non_onc"]; SITES=["haematological","thoracic","gastrointestinal","other_solid"]
CH4=["ecgResp-numLimit-respRate-low","ecg-numLimit-heartRate-high","ecg-numLimit-heartRate-low","spO2-numLimit-satO2-low","ecgResp-numLimit-respRate-high"]
# ---------- Part 1
def blank(): return {"adm":set(),"pat":set(),"days":0.0,"runs":0,"rsec":0,"sil":0,"ina":0,"klass":collections.Counter(),"ch_runs":collections.Counter(),"ch_rsec":collections.Counter(),"ch_sil":collections.Counter()}
S={("tier",t):blank() for t in TIERS}; S.update({("site",s):blank() for s in SITES})
seen=set(); days_done=set()
for fp in sorted(glob.glob(f"{R}/alarm_epidemiology/a1_rows_*.csv")):
    for r in csv.DictReader(open(fp)):
        key=(r["aMRN"],r["aCSN"],r["alarm_base"])
        if key in seen or r["alarm_base"].startswith("__"): continue
        seen.add(key); a=r["aCSN"]
        targets=[("tier",tier.get(a))]+([("site",site[a])] if a in site else [])
        for tg in targets:
            if tg not in S: continue
            d=S[tg]
            if (tg,a) not in days_done:
                days_done.add((tg,a)); d["adm"].add(a); d["pat"].add(r["aMRN"])
                span=(int(r["adm_last_s"])-int(r["adm_first_s"]))/86400 if r["adm_first_s"] else 0
                d["days"]+=max(span,1/24)
            runs=int(r["n_runs"]); rsec=int(r["run_seconds"]); sil=int(r["n_sil"]); ina=int(r["n_inact"])
            d["runs"]+=runs; d["rsec"]+=rsec; d["sil"]+=sil; d["ina"]+=ina; d["klass"][r["klass"]]+=runs
            b=r["alarm_base"]
            if b in CH4: d["ch_runs"][b]+=runs; d["ch_rsec"][b]+=rsec; d["ch_sil"][b]+=sil
part1={}
for (kind,name),d in S.items():
    part1[f"{kind}:{name}"]={"admissions":len(d["adm"]),"patients":len(d["pat"]),"admission_days":round(d["days"],1),
        "runs_per_adm_day":round(d["runs"]/d["days"],1) if d["days"] else None,
        "alarm_hours_per_adm_day":round(d["rsec"]/3600/d["days"],2) if d["days"] else None,
        "silenced_share":round(2*d["sil"]/max(d["rsec"],1),4),"inactivated_share":round(2*d["ina"]/max(d["rsec"],1),4),
        "class_runs_per_adm_day":{k:round(v/d["days"],1) for k,v in d["klass"].items()} if d["days"] else {},
        "channel_runs_per_adm_day":{k:round(d["ch_runs"][k]/d["days"],2) for k in CH4} if d["days"] else {},
        "channel_silenced_share":{k:round(2*d["ch_sil"][k]/max(d["ch_rsec"][k],1),3) for k in CH4}}
# ---------- Part 2 (TEST by tier)
SEL={"ecg-numLimit-heartRate-high":(8,16),"ecg-numLimit-heartRate-low":(0,30),"ecgResp-numLimit-respRate-low":(16,8),"spO2-numLimit-satO2-low":(16,16)}
NAI=(0,30)
agg=collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0]*6)); nadm=collections.defaultdict(set); seen=set()
for fp in sorted(glob.glob(f"{R}/delay_model/a2_TEST_rows_*.csv")):
    for r in csv.DictReader(open(fp)):
        key=(r["aMRN"],r["aCSN"],r["channel"],r["tau"],r["g"])
        if key in seen: continue
        seen.add(key); ch=r["channel"]; cell=(int(r["tau"]),int(r["g"]))
        lab="naive" if cell==NAI else ("selected" if cell==SEL[ch] else None)
        if lab is None: continue
        t=tier.get(r["aCSN"]); g_=site.get(r["aCSN"])
        if t not in TIERS: continue
        strata=[t]+([f"group:{g_}"] if g_ in SITES else [])
        vals=[float(r[c]) for c in ("rec_runs","match","recon_runs","recon_secs","rec_secs","onset_within10")]
        if vals[4]<=0: continue
        for st in strata:
            a=agg[(st,ch,lab)][r["aMRN"]]
            for i in range(6): a[i]+=vals[i]
            nadm[(st,ch,lab)].add(r["aCSN"])
            if cell==NAI and SEL[ch]==NAI:   # HR-low: naive is also the selected cell
                a2=agg[(st,ch,"selected")][r["aMRN"]]
                for i in range(6): a2[i]+=vals[i]
                nadm[(st,ch,"selected")].add(r["aCSN"])
rng=np.random.default_rng(20260831); B=1000; part2={}
for (t,ch,lab),per in sorted(agg.items()):
    M=np.array([per[p] for p in sorted(per)]); n=len(M); tot=M.sum(axis=0)
    rho=tot[3]/tot[4]; idx=rng.integers(0,n,size=(B,n)); rr=M[idx,3].sum(axis=1)/M[idx,4].sum(axis=1)
    lo,hi=np.percentile(rr,[2.5,97.5])
    part2[f"{t}|{ch}|{lab}"]={"n_patients":int(n),"n_admissions":len(nadm[(t,ch,lab)]),"burden_ratio":round(float(rho),3),
        "ci":[round(float(lo),3),round(float(hi),3)],"recall":round(tot[1]/tot[0],3) if tot[0] else None,
        "onset10":round(tot[5]/tot[1],3) if tot[1] else None,"recorded_hours":round(tot[4]/3600,1)}
json.dump({"part1_burden_by_stratum":part1,"part2_TEST_fidelity_by_tier":part2},open(f"{O}/ONCOLOGY_STRATA.json","w"),indent=1)
for k,v in part1.items(): print(k,{kk:vv for kk,vv in v.items() if kk not in ("class_runs_per_adm_day","channel_runs_per_adm_day","channel_silenced_share")})
for k,v in part2.items(): print(k,v["n_patients"],v["n_admissions"],v["burden_ratio"],v["ci"],v["recall"],v["onset10"])
