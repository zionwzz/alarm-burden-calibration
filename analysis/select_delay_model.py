#!/usr/bin/env python3
"""Selection of the annunciation-delay model on the fitting split and its behaviour on the
tuning split. Selection metric: pooled episode-matching F1, ties broken by the burden
ratio closest to one and then by the smaller persistence requirement.
"""
from alarmreplay.paths import WORK as R
import csv,glob,os,json,math,collections,sys
def load(split):
    agg=collections.defaultdict(lambda: collections.Counter()); seen=set()
    for fp in sorted(glob.glob(f"{R}/delay_model/a2_{split}_rows_*.csv")):
        for r in csv.DictReader(open(fp)):
            key=(r["aMRN"],r["aCSN"],r["channel"],r["tau"],r["g"])
            if key in seen: continue
            seen.add(key)
            d=agg[(r["channel"],int(r["tau"]),int(r["g"]))]
            for k2 in ["rec_runs","match","recon_runs","recon_secs","rec_secs","onset_within10"]:
                d[k2]+=int(r[k2])
    return agg
def metrics(d):
    rec=d["rec_runs"]; mt=d["match"]; rc=d["recon_runs"]
    recall=mt/rec if rec else float("nan")
    prec=mt/rc if rc else float("nan")
    f1=2*prec*recall/(prec+recall) if (prec+recall)>0 else 0.0
    br=d["recon_secs"]/d["rec_secs"] if d["rec_secs"] else float("nan")
    on10=d["onset_within10"]/mt if mt else float("nan")
    return recall,prec,f1,br,on10
mode=sys.argv[1] if len(sys.argv)>1 else "select"
fit=load("FIT")
chans=sorted({c for c,_,_ in fit})
sel={}
rows=[["channel","tau","g","FIT_recall","FIT_precision","FIT_F1","FIT_burden_ratio","FIT_onset_within10","selected"]]
for c in chans:
    best=None
    for (cc,tau,g),d in fit.items():
        if cc!=c: continue
        rec,prec,f1,br,on10=metrics(d)
        score=(round(f1,4),-abs(math.log(br)) if br and br>0 else -99,-tau)
        if best is None or score>best[0]: best=(score,(tau,g,rec,prec,f1,br,on10))
    tau,g,rec,prec,f1,br,on10=best[1]
    sel[c]={"tau":tau,"g":g,"FIT":{"recall":round(rec,4),"precision":round(prec,4),"F1":round(f1,4),
            "burden_ratio":round(br,4),"onset_within10":round(on10,4)}}
    for (cc,t2,g2),d in sorted(fit.items()):
        if cc!=c: continue
        r2,p2,f2,b2,o2=metrics(d)
        rows.append([c,t2,g2,round(r2,4),round(p2,4),round(f2,4),round(b2,4),round(o2,4),
                     "YES" if (t2,g2)==(tau,g) else ""])
with open(f"{R}/delay_model/A2_FIT_GRID.csv","w",newline="") as f: csv.writer(f).writerows(rows)
# TUNE check of the selected combos (descriptive; the held-out criteria are fixed in advance)
tune=load("TUNE")
for c in chans:
    tau,g=sel[c]["tau"],sel[c]["g"]
    d=tune.get((c,tau,g))
    if d:
        rec,prec,f1,br,on10=metrics(d)
        sel[c]["TUNE"]={"recall":round(rec,4),"precision":round(prec,4),"F1":round(f1,4),
                        "burden_ratio":round(br,4),"onset_within10":round(on10,4)}
json.dump(sel,open(f"{R}/delay_model/A2_SELECTED_MODEL.json","w"),indent=1)
print(json.dumps(sel,indent=1))
