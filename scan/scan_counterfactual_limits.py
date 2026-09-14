#!/usr/bin/env python3
"""Excursion durations under counterfactual limits. For each admission, channel and limit
offset (0 = the recorded carried-forward limit, positive = looser) the numeric excursions
are recomputed and their duration histogram recorded: sixty 2-second cells below 120 s
and four tail bins with their counts and seconds.

Usage: scan_counterfactual_limits.py <stripe> <n_stripes> <seconds_budget>
"""
from alarmreplay.paths import WORK as R, TELEMETRY as TD, COHORT as SAMP
from alarmreplay.scan_common import parse_time, runs_from, in_any, limit_lookup, crossings, run_overshoots, run_stats, ReadLog, Resumable
import csv,gzip,json,os,sys,time,bisect,collections
stripe,nstr,budget=int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3])
SPLIT="FIT"; GAPS=(4,); NB=60; SUB=4; PHASE=0; TAILB=[120,180,300,600]   # tail bins: [120,180),[180,300),[300,600),[600,inf)
# PD_FULL=1 replays the whole fitting split at the recorded limit only (offset 0), the descriptive
# comparison the representativeness table reads as pd_rows_<stripe>.csv; the same eleven columns
# are written, of which that comparison uses dwell_hist, n_over120 and sum_secs.
FULL=bool(os.environ.get("PD_FULL"))
if FULL: SUB=1
if os.environ.get("ALARM_TRIAL"):
    ST=f"{R}/cache/_trial_pd_state_{stripe}.json"; OUT=f"{R}/cache/_trial_pd_rows_{stripe}.csv"
elif FULL:
    ST=f"{R}/cache/pd_state_{stripe}.json"; OUT=f"{R}/policy/pd_rows_{stripe}.csv"
else:
    ST=f"{R}/cache/pd2_state_{stripe}.json"; OUT=f"{R}/policy/pd2_rows_{stripe}.csv"
RL=ReadLog(ST.replace("_state_","_readlog_").replace(".json",".csv"))
CH={"ecg-numLimit-heartRate-high":("ecg-heartRate#1","high"),
    "ecg-numLimit-heartRate-low":("ecg-heartRate#1","low"),
    "spO2-numLimit-satO2-low":("spO2-satO2#1","low"),
    "ecgResp-numLimit-respRate-low":("ecgResp-respRate#1","low")}
MEAS={v[0] for v in CH.values()}
RANGE={"ecg-heartRate#1":(20,250),"spO2-satO2#1":(50,100),"ecgResp-respRate#1":(4,60)}
# STEP = one unit of limit loosening, in each channel's own units; offsets -1..+3 (positive = looser)
STEP={"ecg-numLimit-heartRate-high":5.0,"ecg-numLimit-heartRate-low":5.0,
      "spO2-numLimit-satO2-low":1.0,"ecgResp-numLimit-respRate-low":1.0}
OFFSETS=[0] if FULL else [-1,0,1,2,3]
splitmap={r["aMRN"]:r["split"] for r in csv.DictReader(open(f"{R}/design/SPLIT_ASSIGNMENT.csv"))}
adms=[(r["aMRN"],r["aCSN"]) for r in csv.DictReader(open(SAMP)) if splitmap.get(r["aMRN"])==SPLIT]
adms=[a for i,a in enumerate(adms) if i%SUB==PHASE]
adms=[a for i,a in enumerate(adms) if i%nstr==stripe]
RS=Resumable(ST,OUT); st=RS.state                      # outputs truncated to the last saved position
t0=time.time(); i=st["i"]; new=RS.is_new(OUT)
fout=RS.files[OUT]; w=csv.writer(fout)
if new:
    w.writerow(["aMRN","aCSN","channel","offset_step","gap_s","n_exc","n_over120","sum_secs","dwell_hist","tail_counts","tail_secs"])
    json.dump({"step_units":STEP,"offsets":OFFSETS,"merge_gaps_s":list(GAPS),"bins":NB,
               "note":"positive offset = looser limit (higher for 'high' channels, lower for 'low')"},
              open(OUT.replace(".csv","_meta.json"),"w"),indent=1)
def save(): RS.save(i)
while i<len(adms) and time.time()-t0<budget:
    m,a=adms[i]; i+=1
    ap=f"{TD}/{m}/{a}/ALARMp{m}e{a}.csv.gz"; mp=f"{TD}/{m}/{a}/MEASUREMENTp{m}e{a}.csv.gz"
    if not (os.path.exists(ap) and os.path.exists(mp)): RL.note(m,a,"missing_file"); continue
    silt=collections.defaultdict(list); lims=collections.defaultdict(list); present=set()
    try:
        with gzip.open(ap,"rt") as f:
            r0=csv.reader(f); next(r0,None)
            for row in r0:
                if len(row)<8: continue
                base=row[1].split("#")[0]
                if base not in CH: continue
                t=parse_time(row[0]); present.add(base)
                if row[4]=="True" or row[3]!="enabled": silt[base].append(t)
                lv=row[5] if row[5] not in ("None","") else None
                hv=row[6] if row[6] not in ("None","") else None
                lims[base].append((t,lv,hv))
    except Exception as ex: RL.note(m,a,"alarm_read_error:"+type(ex).__name__); continue
    if not present:
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
        if base not in present or mch not in vals: continue
        vv=sorted(vals[mch])
        sil=runs_from(sorted(silt[base]),30); silidx=[s for s,_ in sil]
        L=sorted(lims[base]); idx=2 if dirn=="high" else 1
        lim_at=limit_lookup(L,idx)
        step=STEP[base]
        for off in OFFSETS:
            shift=off*step*(1.0 if dirn=="high" else -1.0)   # looser = higher for high, lower for low
            cross,_=crossings(vv,sil,silidx,lim_at,dirn,shift)
            for gp in GAPS:
                rr=runs_from(cross,gp)
                if not rr: continue
                hist=[0]*NB; over=0; tot=0; tc=[0]*len(TAILB); ts_=[0]*len(TAILB)
                for s,e in rr:
                    d=e-s+2; tot+=d
                    k=d//2
                    if k<NB: hist[k]+=1
                    else:
                        over+=1; b=bisect.bisect_right(TAILB,d)-1; tc[b]+=1; ts_[b]+=d
                w.writerow([m,a,base,off,gp,len(rr),over,tot,"|".join(map(str,hist)),"|".join(map(str,tc)),"|".join(map(str,ts_))])
    if i%10==0: save()
save(); fout.close()
print(f"{'DONE' if i>=len(adms) else 'PARTIAL'} {SPLIT} stripe={stripe}/{nstr} {i}/{len(adms)} elapsed={time.time()-t0:.0f}s rate={i/max(time.time()-t0,0.1):.2f}/s")
