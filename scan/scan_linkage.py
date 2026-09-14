#!/usr/bin/env python3
"""Linkage of replayed excursions to recorded annunciations. Excursions and annunciation
runs are merged at the same gap, silenced and inactivated samples are removed from both
sides, and every annunciation run is attached to the excursion it overlaps most.
One row per excursion: duration, linkage indicator, onset lag, extension, attached
seconds and number of attached runs; per-admission totals under two merge conventions.

Usage: scan_linkage.py <stripe> <n_stripes> <seconds_budget>
"""
from alarmreplay.paths import WORK as R, TELEMETRY as TD, COHORT as SAMP
from alarmreplay.scan_common import parse_time, runs_from, in_any, limit_lookup, crossings, run_overshoots, run_stats, ReadLog, Resumable
import csv,gzip,json,os,sys,time,bisect,collections
stripe,nstr,budget=int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3])
SPLIT="FIT"; EGAP=4; SILGAP=30; SUB=4; BIG=30; W=30
ST=f"{R}/cache/lk_state_{stripe}.json"; OUT=f"{R}/policy/lk_rows_{stripe}.csv"; OUTT=f"{R}/policy/lk_tot_{stripe}.csv"
RL=ReadLog(ST.replace("_state_","_readlog_").replace(".json",".csv"))
CH={"ecg-numLimit-heartRate-high":("ecg-heartRate#1","high"),"ecg-numLimit-heartRate-low":("ecg-heartRate#1","low"),
    "spO2-numLimit-satO2-low":("spO2-satO2#1","low"),"ecgResp-numLimit-respRate-low":("ecgResp-respRate#1","low")}
MEAS={v[0] for v in CH.values()}
RANGE={"ecg-heartRate#1":(20,250),"spO2-satO2#1":(50,100),"ecgResp-respRate#1":(4,60)}
splitmap={r["aMRN"]:r["split"] for r in csv.DictReader(open(f"{R}/design/SPLIT_ASSIGNMENT.csv"))}
adms=[(r["aMRN"],r["aCSN"]) for r in csv.DictReader(open(SAMP)) if splitmap.get(r["aMRN"])==SPLIT]
adms=[a for i,a in enumerate(adms) if i%SUB==0]
adms=[a for i,a in enumerate(adms) if i%nstr==stripe]
RS=Resumable(ST,OUT,OUTT); st=RS.state                 # outputs truncated to the last saved position
t0=time.time(); i=st["i"]; new=RS.is_new(OUT)
fout=RS.files[OUT]; w=csv.writer(fout); ftot=RS.files[OUTT]; wt=csv.writer(ftot)
if new:
    w.writerow(["aMRN","aCSN","channel","D_s","A","L_s","E_s","S_s","K"])
    wt.writerow(["aMRN","aCSN","channel","exc4","ann4","unl4","nexc4","nann4","exc30","ann30","unl30"])
def save(): RS.save(i)
def link(exc,ann):
    """attach every annunciation run to the excursion of greatest positive overlap; return per-excursion lists and unlinked seconds"""
    att=[[] for _ in exc]; unl=0; estart=[s for s,_ in exc]
    for As,Ae in ann:
        k=bisect.bisect_left(estart,As-W); best=None; bov=0
        while k<len(exc) and exc[k][0]<=Ae:
            ov=min(Ae,exc[k][1])-max(As,exc[k][0])
            if ov>bov: bov=ov; best=k
            k+=1
        if best is None or bov<=0: unl+=Ae-As+2
        else: att[best].append((As,Ae))
    return att,unl
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
                t=parse_time(row[0])
                sil=(row[4]=="True" or row[3]!="enabled")
                if sil: silt[base].append(t)
                else: rec[base].append(t)                       # un-silenced alarm rows only
                lv=row[5] if row[5] not in ("None","") else None
                hv=row[6] if row[6] not in ("None","") else None
                lims[base].append((t,lv,hv))
    except Exception as ex: RL.note(m,a,"alarm_read_error:"+type(ex).__name__); continue
    if not rec and not silt:
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
        if mch not in vals or (base not in rec and base not in silt): continue
        vv=sorted(vals[mch])
        sil=runs_from(sorted(silt[base]),SILGAP); silidx=[s for s,_ in sil]
        L=sorted(lims[base]); idx=2 if dirn=="high" else 1
        lim_at=limit_lookup(L,idx)
        cross,_=crossings(vv,sil,silidx,lim_at,dirn)
        rt=sorted(rec.get(base,[]))
        exc4=runs_from(cross,EGAP); ann4=runs_from(rt,EGAP)
        exc30=runs_from(cross,BIG); ann30=runs_from(rt,BIG)
        att,unl4=link(exc4,ann4); _,unl30=link(exc30,ann30)
        for (s,e),runs in zip(exc4,att):
            D=e-s+2
            if not runs: w.writerow([m,a,base,D,0,"","","",0]); continue
            runs.sort(); As=runs[0][0]; Ae=max(r[1] for r in runs); S=sum(r[1]-r[0]+2 for r in runs)
            w.writerow([m,a,base,D,1,As-s,Ae-e,S,len(runs)])
        wt.writerow([m,a,base,sum(e-s+2 for s,e in exc4),sum(e-s+2 for s,e in ann4),unl4,len(exc4),len(ann4),
                     sum(e-s+2 for s,e in exc30),sum(e-s+2 for s,e in ann30),unl30])
    if i%10==0: save()
save(); fout.close(); ftot.close()
print(f"{'DONE' if i>=len(adms) else 'PARTIAL'} {SPLIT} stripe={stripe}/{nstr} {i}/{len(adms)} elapsed={time.time()-t0:.0f}s")
