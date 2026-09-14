#!/usr/bin/env python3
"""Distribution of the interval between a numeric crossing and the recorded annunciation,
on the fitting split at the naive setting.

Usage: scan_onset_lead.py <stripe> <n_stripes> <seconds_budget>
"""
from alarmreplay.paths import WORK as R, TELEMETRY as TD, COHORT as SAMP
from alarmreplay.scan_common import parse_time, runs_from, in_any, limit_lookup, crossings, run_overshoots, run_stats, ReadLog, Resumable
import csv,gzip,json,os,sys,time,bisect,collections
stripe,nstr,budget=int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3])
SPLIT="FIT"; TAU=0; G=30; GAP=30; LO=-120; HI=120; BW=2
ST=f"{R}/cache/c1_state_{stripe}_{nstr}.json"
OUT=f"{R}/comparison/c1_onset_hist_{stripe}_{nstr}.json"
RL=ReadLog(ST.replace("_state_","_readlog_").replace(".json",".csv"))
CH={"ecg-numLimit-heartRate-high":("ecg-heartRate#1","high"),
    "ecg-numLimit-heartRate-low":("ecg-heartRate#1","low"),
    "spO2-numLimit-satO2-low":("spO2-satO2#1","low"),
    "ecgResp-numLimit-respRate-low":("ecgResp-respRate#1","low")}
MEAS={v[0] for v in CH.values()}
RANGE={"ecg-heartRate#1":(20,250),"spO2-satO2#1":(50,100),"ecgResp-respRate#1":(4,60)}
NB=(HI-LO)//BW
splitmap={r["aMRN"]:r["split"] for r in csv.DictReader(open(f"{R}/design/SPLIT_ASSIGNMENT.csv"))}
adms=[(r["aMRN"],r["aCSN"]) for r in csv.DictReader(open(SAMP)) if splitmap.get(r["aMRN"])==SPLIT]
adms=[a for i,a in enumerate(adms) if i%nstr==stripe]
if os.path.exists(ST):
    st=json.load(open(ST))
else:
    st={"i":0,"hist":{b:[0]*NB for b in CH},"lo_tail":{b:0 for b in CH},"hi_tail":{b:0 for b in CH},"n":{b:0 for b in CH}}
i=st["i"]; hist=st["hist"]; lo_tail=st["lo_tail"]; hi_tail=st["hi_tail"]; ntot=st["n"]
t0=time.time()
def save():
    json.dump({"i":i,"hist":hist,"lo_tail":lo_tail,"hi_tail":hi_tail,"n":ntot},open(ST+".tmp","w"))
    os.replace(ST+".tmp",ST)
    json.dump({"split":SPLIT,"tau":TAU,"g":G,"bin_width_s":BW,"lo":LO,"hi":HI,
               "hist":hist,"lo_tail":lo_tail,"hi_tail":hi_tail,"n_matched":ntot,
               "admissions_done":i,"admissions_total":len(adms)},open(OUT+".tmp","w"))
    os.replace(OUT+".tmp",OUT)
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
        vv=sorted(vals[mch])
        sil=runs_from(sorted(silt[base]),GAP); silidx=[s for s,_ in sil]
        L=sorted(lims[base])
        idx=2 if dirn=="high" else 1
        lim_at=limit_lookup(L,idx)
        cross,_=crossings(vv,sil,silidx,lim_at,dirn)
        rec_sc=[(s,e) for s,e in runs_from(sorted(rec[base]),GAP) if not in_any(s,sil,silidx)]
        rr=runs_from(cross,G)
        starts=[s for s,_ in rr]
        for s,e in rec_sc:
            k=bisect.bisect_left(starts,s-GAP)
            while k<len(rr) and rr[k][0]<=e+GAP:
                if rr[k][1]>=s-GAP:
                    d=s-rr[k][0]
                    ntot[base]+=1
                    if d<LO: lo_tail[base]+=1
                    elif d>=HI: hi_tail[base]+=1
                    else: hist[base][(d-LO)//BW]+=1
                    break
                k+=1
    if i%10==0: save()
save()
print(f"{'DONE' if i>=len(adms) else 'PARTIAL'} c1 stripe={stripe}/{nstr} {i}/{len(adms)} elapsed={time.time()-t0:.0f}s")
