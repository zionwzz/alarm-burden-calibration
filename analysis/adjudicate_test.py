#!/usr/bin/env python3
"""Adjudication of the selected delay model against the adequacy criteria, fixed before the
held-out data were examined, on
the held-out split, and the naive-versus-selected burden comparison. Run once.
"""
from alarmreplay.paths import WORK as R
import csv,glob,os,json,math,collections,random
LOG=f"{R}/comparison/TEST_ADJUDICATED.log"
# The held-out split is read once. A repeat run on the same rows requires a stated reason
# (ADJUDICATE_RERUN_REASON); every earlier result is kept beside the new one and the reason is
# appended to the log, so no held-out result is ever overwritten or lost.
if os.path.exists(LOG):
    reason=os.environ.get("ADJUDICATE_RERUN_REASON","").strip()
    assert reason, "the held-out split has already been adjudicated; a repeat run needs ADJUDICATE_RERUN_REASON"
    import datetime as _dt, shutil as _sh
    n=1+sum(1 for l in open(LOG) if l.startswith("repeat"))
    for fp in (f"{R}/delay_model/A2_TEST_ADJUDICATION.csv",f"{R}/comparison/A3_NAIVE_VS_DELAY_AWARE.csv",f"{R}/delay_model/A2_TEST_RESULTS.json"):
        if os.path.exists(fp): _sh.copy(fp, fp.replace(".csv",f".previous{n}.csv").replace(".json",f".previous{n}.json"))
    open(LOG,"a").write(f"repeat {n} {_dt.datetime.utcnow().isoformat()}Z: {reason}\n")
sel=json.load(open(f"{R}/delay_model/A2_SELECTED_MODEL.json"))
crit=json.load(open(f"{R}/design/DESIGN.json"))["validation_criteria_TEST"]
per=collections.defaultdict(lambda: collections.defaultdict(collections.Counter)) # (channel,tau,g)->patient->counters
seen=set()
for fp in sorted(glob.glob(f"{R}/delay_model/a2_TEST_rows_*.csv")):
    for r in csv.DictReader(open(fp)):
        key=(r["aMRN"],r["aCSN"],r["channel"],r["tau"],r["g"])
        if key in seen: continue
        seen.add(key)
        d=per[(r["channel"],int(r["tau"]),int(r["g"]))][r["aMRN"]]
        for k2 in ["rec_runs","match","recon_runs","recon_secs","rec_secs","onset_within10"]:
            d[k2]+=int(r[k2])
def pool(pats):
    t=collections.Counter()
    for d in pats.values():
        for k,v in d.items(): t[k]+=v
    return t
def metrics(d):
    rec=d["rec_runs"]; mt=d["match"]; rc=d["recon_runs"]
    recall=mt/rec if rec else float("nan"); prec=mt/rc if rc else float("nan")
    br=d["recon_secs"]/d["rec_secs"] if d["rec_secs"] else float("nan")
    on10=d["onset_within10"]/mt if mt else float("nan")
    return recall,prec,br,on10
rng=random.Random(20260831)
def boot_ci(pats,fn,B=400):
    keys=sorted(pats); vals=[]          # sorted, so the draws do not depend on the order rows were written in
    for _ in range(B):
        t=collections.Counter()
        for k in [keys[rng.randrange(len(keys))] for _ in keys]:
            for k2,v in pats[k].items(): t[k2]+=v
        vals.append(fn(t))
    vals=[v for v in vals if v==v]
    vals.sort()
    return (vals[int(0.025*len(vals))],vals[int(0.975*len(vals))]) if vals else (float("nan"),)*2
res={"criteria":crit,"channels":{}}
rows=[["channel","tau","g","TEST_recall","recall_CI","TEST_burden_ratio","br_CI","TEST_onset_within10","criterion_recall>=0.90","criterion_br_in_[0.8,1.25]","criterion_onset>=0.80","RETAINED"]]
a3=[["channel","naive_burden_ratio","naive_CI","delay_aware_burden_ratio","delay_CI","overcount_factor"]]
for c,s in sel.items():
    tau,g=s["tau"],s["g"]
    pats=per.get((c,tau,g),{})
    if not pats: continue
    t=pool(pats); rec,prec,br,on10=metrics(t)
    ci_r=boot_ci(pats,lambda d:(d["match"]/d["rec_runs"]) if d["rec_runs"] else float("nan"))
    ci_b=boot_ci(pats,lambda d:(d["recon_secs"]/d["rec_secs"]) if d["rec_secs"] else float("nan"))
    g1=rec>=crit["episode_recall_min"]; g2=crit["burden_ratio_range"][0]<=br<=crit["burden_ratio_range"][1]; g3=on10>=crit["onset_within_10s_min"]
    keep=g1 and g2 and g3
    res["channels"][c]={"tau":tau,"g":g,"TEST":{"recall":round(rec,4),"precision":round(prec,4),
        "burden_ratio":round(br,4),"onset_within10":round(on10,4)},
        "recall_CI":[round(x,4) for x in ci_r],"burden_CI":[round(x,4) for x in ci_b],
        "criteria":{"recall":g1,"burden":g2,"onset":g3},"RETAINED":keep}
    rows.append([c,tau,g,round(rec,4),f"[{ci_r[0]:.3f},{ci_r[1]:.3f}]",round(br,4),
                 f"[{ci_b[0]:.3f},{ci_b[1]:.3f}]",round(on10,4),g1,g2,g3,keep])
    # A3 naive comparator = (tau=0, g=30)
    pn=per.get((c,0,30),{})
    if pn:
        tn=pool(pn); _,_,brn,_=metrics(tn)
        cin=boot_ci(pn,lambda d:(d["recon_secs"]/d["rec_secs"]) if d["rec_secs"] else float("nan"))
        a3.append([c,round(brn,4),f"[{cin[0]:.3f},{cin[1]:.3f}]",round(br,4),
                   f"[{ci_b[0]:.3f},{ci_b[1]:.3f}]",round(brn/br,3) if br else ""])
os.makedirs(f"{R}/comparison",exist_ok=True)
with open(f"{R}/delay_model/A2_TEST_ADJUDICATION.csv","w",newline="") as f: csv.writer(f).writerows(rows)
with open(f"{R}/comparison/A3_NAIVE_VS_DELAY_AWARE.csv","w",newline="") as f: csv.writer(f).writerows(a3)
json.dump(res,open(f"{R}/delay_model/A2_TEST_RESULTS.json","w"),indent=1)
open(LOG,"a").write("adjudicated once\n") if not os.path.exists(LOG) else None
print(json.dumps(res,indent=1)[:2000])
