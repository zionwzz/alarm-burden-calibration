#!/usr/bin/env python3
"""Annunciation-delay model scan at the native 2-second resolution. For each admission,
channel and candidate (persistence tau, merge gap g): recorded runs, matched runs,
reconstructed runs and seconds, and onset agreement.

Usage: scan_delay_model.py <stripe> <n_stripes> <seconds_budget>
"""
from alarmreplay.paths import WORK as R, TELEMETRY as TD, COHORT as SAMP
from alarmreplay.scan_common import parse_time, runs_from, in_any, limit_lookup, crossings, run_overshoots, run_stats, ReadLog, Resumable
import csv,gzip,json,os,sys,time,bisect,collections
split,stripe,nstr,budget=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),float(sys.argv[4])
ST=f"{R}/cache/a2_{split}_state_{stripe}.json"
OUT=f"{R}/delay_model/a2_{split}_rows_{stripe}.csv"
RL=ReadLog(ST.replace("_state_","_readlog_").replace(".json",".csv"))
GAP=30; TAUS=[0,4,8,12,16,24]; GS=[8,16,30]
CH={"ecg-numLimit-heartRate-high":("ecg-heartRate#1","high"),
    "ecg-numLimit-heartRate-low":("ecg-heartRate#1","low"),
    "spO2-numLimit-satO2-low":("spO2-satO2#1","low"),
    "ecgResp-numLimit-respRate-low":("ecgResp-respRate#1","low")}
MEAS={v[0] for v in CH.values()}
RANGE={"ecg-heartRate#1":(20,250),"spO2-satO2#1":(50,100),"ecgResp-respRate#1":(4,60)}
splitmap={r["aMRN"]:r["split"] for r in csv.DictReader(open(f"{R}/design/SPLIT_ASSIGNMENT.csv"))}
adms=[(r["aMRN"],r["aCSN"]) for r in csv.DictReader(open(SAMP)) if splitmap.get(r["aMRN"])==split]
adms=[a for i,a in enumerate(adms) if i%nstr==stripe]
RS=Resumable(ST,OUT); st=RS.state                      # output truncated to the last saved position
t0=time.time(); i=st["i"]; new=RS.is_new(OUT)
fout=RS.files[OUT]; w=csv.writer(fout)
if new: w.writerow(["aMRN","aCSN","channel","tau","g","rec_runs","match","recon_runs","recon_secs","rec_secs","onset_within10"])
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
        if i%10==0: RS.save(i)
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
        vv=sorted(vals[mch])
        sil=runs_from(sorted(silt[base]),GAP); silidx=[s for s,_ in sil]
        L=sorted(lims[base])
        idx=2 if dirn=="high" else 1
        lim_at=limit_lookup(L,idx)
        cross,_=crossings(vv,sil,silidx,lim_at,dirn)
        rec_sc=[(s,e) for s,e in runs_from(sorted(rec[base]),GAP) if not in_any(s,sil,silidx)]
        rec_secs=sum(e-s+2 for s,e in rec_sc)
        for g in GS:
            cr=runs_from(cross,g)
            for tau in TAUS:
                rr=[(s,e) for s,e in cr if e-s+2>=tau]
                rsec=sum(e-s+2 for s,e in rr)
                starts=[s for s,_ in rr]
                match=on10=0
                for s,e in rec_sc:
                    k=bisect.bisect_left(starts,s-GAP)
                    while k<len(rr) and rr[k][0]<=e+GAP:
                        if rr[k][1]>=s-GAP:
                            match+=1
                            if abs(rr[k][0]+tau-s)<=10: on10+=1
                            break
                        k+=1
                w.writerow([m,a,base,tau,g,len(rec_sc),match,len(rr),rsec,rec_secs,on10])
    if i%10==0: RS.save(i)
RS.save(i); RS.close()
print(f"{'DONE' if i>=len(adms) else 'PARTIAL'} {split} stripe={stripe}/{nstr} {i}/{len(adms)} elapsed={time.time()-t0:.0f}s rate={i/max(time.time()-t0,0.1):.2f}/s")
