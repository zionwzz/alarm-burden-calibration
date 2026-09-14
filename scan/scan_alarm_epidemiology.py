#!/usr/bin/env python3
"""Recorded alarm burden and silencing, one row per admission and alarm state.
Reads the alarm stream only. Stripeable and resumable.

Usage: scan_alarm_epidemiology.py <stripe> <n_stripes> <seconds_budget>
"""
from alarmreplay.paths import WORK as R, TELEMETRY as TD, COHORT as SAMP
from alarmreplay.scan_common import parse_time, runs_from, in_any, limit_lookup, crossings, run_overshoots, run_stats, ReadLog, Resumable
import csv,gzip,json,os,sys,time,collections
stripe,nstr,budget=int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3])
ST=f"{R}/cache/a1_state_{stripe}.json"
OUT=f"{R}/alarm_epidemiology/a1_rows_{stripe}.csv"
GAP=30
def klass(nm):
    if "-numLimit-" in nm: return "numLimit"
    if "-rhythm-" in nm: return "rhythm"
    if "-tech-" in nm: return "tech"
    if nm.startswith("system-"): return "system"
    return "other"
adms=[]
with open(SAMP) as f:
    for i,row in enumerate(csv.DictReader(f)):
        if i%nstr==stripe: adms.append((row["aMRN"],row["aCSN"]))
RS=Resumable(ST,OUT); st=RS.state                      # output truncated to the last saved position
t0=time.time(); i=st["i"]; wrote_header=not RS.is_new(OUT)
fout=RS.files[OUT]; w=csv.writer(fout)
if not wrote_header:
    w.writerow(["aMRN","aCSN","alarm_base","klass","n_rows","n_sil","n_inact","n_runs","run_seconds",
                "n_distinct_setLow","n_distinct_setHigh","modal_setLow","modal_setHigh",
                "adm_first_s","adm_last_s"])
while i<len(adms) and time.time()-t0<budget:
    m,a=adms[i]; i+=1
    p=f"{TD}/{m}/{a}/ALARMp{m}e{a}.csv.gz"
    if not os.path.exists(p):
        w.writerow([m,a,"__NO_ALARM_FILE__","none",0,0,0,0,0,0,0,"","","",""]); continue
    per=collections.defaultdict(lambda:{"n":0,"sil":0,"ina":0,"times":[],"lo":collections.Counter(),"hi":collections.Counter()})
    lo_all=hi_all=None
    try:
        with gzip.open(p,"rt") as f:
            r=csv.reader(f); next(r,None)
            for row in r:
                if len(row)<8: continue
                base=row[1].split("#")[0]
                d=per[base]; d["n"]+=1
                if row[4]=="True": d["sil"]+=1
                if row[3]!="enabled": d["ina"]+=1
                d["times"].append(parse_time(row[0]))
                if row[5] not in ("None",""): d["lo"][row[5]]+=1
                if row[6] not in ("None",""): d["hi"][row[6]]+=1
    except Exception:
        w.writerow([m,a,"__READ_ERROR__","none",0,0,0,0,0,0,0,"","","",""]); 
        RS.save(i); continue
    fs=min((min(d["times"]) for d in per.values() if d["times"]),default=0)
    ls=max((max(d["times"]) for d in per.values() if d["times"]),default=0)
    for base,d in per.items():
        rr=runs_from(sorted(d["times"]),GAP); runs=len(rr); rsec=sum(e-s+2 for s,e in rr)
        w.writerow([m,a,base,klass(base),d["n"],d["sil"],d["ina"],runs,rsec,
                    len(d["lo"]),len(d["hi"]),
                    (d["lo"].most_common(1)[0][0] if d["lo"] else ""),
                    (d["hi"].most_common(1)[0][0] if d["hi"] else ""),fs,ls])
    if i%25==0: RS.save(i)
RS.save(i); RS.close()
print(f"{'DONE' if i>=len(adms) else 'PARTIAL'} stripe={stripe}/{nstr} {i}/{len(adms)} elapsed={time.time()-t0:.0f}s rate={i/max(time.time()-t0,1):.2f}/s")
