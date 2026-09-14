#!/usr/bin/env python3
"""Measurement-convention scan on the held-out split: the fixed quantities re-derived
under perturbations of the merge gap, the silencing handling, the limit convention, the
matching tolerance and the onset window, one at a time from the fixed defaults.

Usage: scan_measurement_conventions.py <stripe> <n_stripes> <seconds_budget>
"""
from alarmreplay.paths import WORK as R, TELEMETRY as TD, COHORT as SAMP
from alarmreplay.scan_common import parse_time, runs_from, in_any, limit_lookup, crossings, run_overshoots, run_stats, ReadLog, Resumable
import csv,gzip,json,os,sys,time,bisect,collections,glob
stripe,nstr,budget=int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3])
SPLIT="TEST"
if os.environ.get("ALARM_TRIAL"):
    ST=f"{R}/cache/_trial_s2_state_{stripe}.json"; OUT=f"{R}/cache/_trial_s2_rows_{stripe}.csv"
else:
    ST=f"{R}/cache/s2_state_{stripe}.json"; OUT=f"{R}/sensitivity/s2_rows_{stripe}.csv"
RL=ReadLog(ST.replace("_state_","_readlog_").replace(".json",".csv"))
SILGAP=30                      # silencing-run merge gap: fixed (not a reported convention)
NAI=(0,30)
SEL={"ecg-numLimit-heartRate-high":(8,16),"ecg-numLimit-heartRate-low":(0,30),
     "ecgResp-numLimit-respRate-low":(16,8),"spO2-numLimit-satO2-low":(16,16)}
CONFIGS=collections.OrderedDict([
 ("default",    dict(d0=30,tol=30,onw=10,sil="exclude",lim="carried")),
 ("d0_8",       dict(d0=8, tol=30,onw=10,sil="exclude",lim="carried")),
 ("d0_16",      dict(d0=16,tol=30,onw=10,sil="exclude",lim="carried")),
 ("d0_60",      dict(d0=60,tol=30,onw=10,sil="exclude",lim="carried")),
 ("tol_10",     dict(d0=30,tol=10,onw=10,sil="exclude",lim="carried")),
 ("tol_20",     dict(d0=30,tol=20,onw=10,sil="exclude",lim="carried")),
 ("tol_60",     dict(d0=30,tol=60,onw=10,sil="exclude",lim="carried")),
 ("onset_6",    dict(d0=30,tol=30,onw=6, sil="exclude",lim="carried")),
 ("onset_20",   dict(d0=30,tol=30,onw=20,sil="exclude",lim="carried")),
 ("sil_include",dict(d0=30,tol=30,onw=10,sil="include",lim="carried")),
 ("lim_modal",  dict(d0=30,tol=30,onw=10,sil="exclude",lim="modal")),
])
CH={"ecg-numLimit-heartRate-high":("ecg-heartRate#1","high"),
    "ecg-numLimit-heartRate-low":("ecg-heartRate#1","low"),
    "spO2-numLimit-satO2-low":("spO2-satO2#1","low"),
    "ecgResp-numLimit-respRate-low":("ecgResp-respRate#1","low")}
MEAS={v[0] for v in CH.values()}
RANGE={"ecg-heartRate#1":(20,250),"spO2-satO2#1":(50,100),"ecgResp-respRate#1":(4,60)}
# ---- cohort modal limits per channel, from the A1 rows (no new I/O on telemetry)
MODAL={}
cnt=collections.defaultdict(collections.Counter)
for fp in sorted(glob.glob(f"{R}/alarm_epidemiology/a1_rows_*.csv")):
    for r in csv.DictReader(open(fp)):
        b=r["alarm_base"]
        if b not in CH: continue
        v=r["modal_setHigh"] if CH[b][1]=="high" else r["modal_setLow"]
        if v in ("","None",None): continue
        try: cnt[b][float(v)]+=1
        except ValueError: pass
for b in CH:
    MODAL[b]=cnt[b].most_common(1)[0][0] if cnt[b] else None
splitmap={r["aMRN"]:r["split"] for r in csv.DictReader(open(f"{R}/design/SPLIT_ASSIGNMENT.csv"))}
adms=[(r["aMRN"],r["aCSN"]) for r in csv.DictReader(open(SAMP)) if splitmap.get(r["aMRN"])==SPLIT]
adms=[a for i,a in enumerate(adms) if i%nstr==stripe]
RS=Resumable(ST,OUT); st=RS.state                      # outputs truncated to the last saved position
t0=time.time(); i=st["i"]; new=RS.is_new(OUT)
fout=RS.files[OUT]; w=csv.writer(fout)
if new:
    w.writerow(["aMRN","aCSN","channel","config","cell","rec_runs","match","recon_runs","recon_secs","rec_secs","onset_within"])
    json.dump({"modal_limits":MODAL,"configs":CONFIGS},open(OUT.replace(".csv","_meta.json"),"w"),indent=1)
def save(): RS.save(i)
while i<len(adms) and time.time()-t0<budget:
    m,a=adms[i]; i+=1
    ap=f"{TD}/{m}/{a}/ALARMp{m}e{a}.csv.gz"; mp=f"{TD}/{m}/{a}/MEASUREMENTp{m}e{a}.csv.gz"
    if not (os.path.exists(ap) and os.path.exists(mp)): RL.note(m,a,"missing_file"); continue
    rec=collections.defaultdict(list); silt=collections.defaultdict(list); lims=collections.defaultdict(list)
    try:
        with gzip.open(ap,"rt") as f:
            r0=csv.reader(f); next(r0,None)
            for row in r0:
                if len(row)<8: continue
                base=row[1].split("#")[0]
                if base not in CH: continue
                t=parse_time(row[0]); rec[base].append(t)
                if row[4]=="True" or row[3]!="enabled": silt[base].append(t)
                lv=row[5] if row[5] not in ("None","") else None
                hv=row[6] if row[6] not in ("None","") else None
                lims[base].append((t,lv,hv))
    except Exception as ex: RL.note(m,a,"alarm_read_error:"+type(ex).__name__); continue
    if not rec:
        if i%10==0: save()
        continue
    vals=collections.defaultdict(list)
    try:
        with gzip.open(mp,"rt") as f:
            next(f,None)
            for line in f:
                p=line.rstrip("\n").split(",")
                if len(p)<5 or p[1] not in MEAS: continue
                try: v=float(p[4])
                except ValueError: continue
                lo,hi=RANGE[p[1]]
                if lo<=v<=hi: vals[p[1]].append((parse_time(p[0]),v))
    except Exception as ex: RL.note(m,a,"measurement_read_error:"+type(ex).__name__); continue
    for base,(mch,dirn) in CH.items():
        if base not in rec or mch not in vals: continue
        vv=sorted(vals[mch]); rect=sorted(rec[base])
        sil=runs_from(sorted(silt[base]),SILGAP); silidx=[s for s,_ in sil]
        L=sorted(lims[base])
        idx=2 if dirn=="high" else 1
        lim_at=limit_lookup(L,idx)
        cross_cache={}; rec_cache={}
        def crossings(silmode,limmode):
            key=(silmode,limmode)
            if key in cross_cache: return cross_cache[key]
            out=[]; modal=MODAL.get(base)
            for t,v in vv:
                if silmode=="exclude" and in_any(t,sil,silidx): continue
                th=modal if limmode=="modal" else lim_at(t)
                if th is None: continue
                if (v>=th) if dirn=="high" else (v<=th): out.append(t)
            cross_cache[key]=out; return out
        def recorded(d0,silmode):
            key=(d0,silmode)
            if key in rec_cache: return rec_cache[key]
            rs=runs_from(rect,d0)
            if silmode=="exclude": rs=[(s,e) for s,e in rs if not in_any(s,sil,silidx)]
            rec_cache[key]=rs; return rs
        for cname,cfg in CONFIGS.items():
            cross=crossings(cfg["sil"],cfg["lim"]); rec_sc=recorded(cfg["d0"],cfg["sil"])
            rec_secs=sum(e-s+2 for s,e in rec_sc); tol=cfg["tol"]; onw=cfg["onw"]
            for cell_label,(tau,g) in (("naive",NAI),("selected",SEL[base])):
                cr=runs_from(cross,g)
                rr=[(s,e) for s,e in cr if e-s+2>=tau]
                rsec=sum(e-s+2 for s,e in rr)
                starts=[s for s,_ in rr]
                match=onw_hit=0
                for s,e in rec_sc:
                    k=bisect.bisect_left(starts,s-tol)
                    while k<len(rr) and rr[k][0]<=e+tol:
                        if rr[k][1]>=s-tol:
                            match+=1
                            if abs(rr[k][0]+tau-s)<=onw: onw_hit+=1
                            break
                        k+=1
                w.writerow([m,a,base,cname,cell_label,len(rec_sc),match,len(rr),rsec,rec_secs,onw_hit])
    if i%10==0: save()
save(); fout.close()
print(f"{'DONE' if i>=len(adms) else 'PARTIAL'} {SPLIT} stripe={stripe}/{nstr} {i}/{len(adms)} elapsed={time.time()-t0:.0f}s rate={i/max(time.time()-t0,0.1):.2f}/s modal={MODAL}")
