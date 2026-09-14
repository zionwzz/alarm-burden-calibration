#!/usr/bin/env python3
"""Linkage scan with excursion features: the maximal and mean excess beyond the limit in
force, the entering slope, the pre-excursion level, dropped samples, and the position of
the excursion relative to recorded within-admission limit changes. Also writes the
limit-change events themselves.

Usage: scan_linkage_features.py <split> <subsample> <phase> <stripe> <n_stripes> <seconds_budget>
"""
from alarmreplay.paths import WORK as R, TELEMETRY as TD, COHORT as SAMP
from alarmreplay.scan_common import parse_time, runs_from, in_any, limit_lookup, crossings, run_overshoots, run_stats, ReadLog, Resumable
import csv,gzip,json,os,sys,time,bisect,collections
SPLIT=sys.argv[1]; SUB=int(sys.argv[2]); PHASE=int(sys.argv[3]); stripe,nstr,budget=int(sys.argv[4]),int(sys.argv[5]),float(sys.argv[6])
EGAP=4; SILGAP=30; BIG=30; W=30; TAG=f"{SPLIT}{SUB}p{PHASE}"
ST=f"{R}/cache/lk2_state_{TAG}_{stripe}.json"; OUT=f"{R}/policy/lk2_rows_{TAG}_{stripe}.csv"; OUTT=f"{R}/policy/lk2_tot_{TAG}_{stripe}.csv"; OUTC=f"{R}/policy/lk2_chg_{TAG}_{stripe}.csv"
RL=ReadLog(ST.replace("_state_","_readlog_").replace(".json",".csv"))
CH={"ecg-numLimit-heartRate-high":("ecg-heartRate#1","high"),"ecg-numLimit-heartRate-low":("ecg-heartRate#1","low"),
    "spO2-numLimit-satO2-low":("spO2-satO2#1","low"),"ecgResp-numLimit-respRate-low":("ecgResp-respRate#1","low")}
MEAS={v[0] for v in CH.values()}
RANGE={"ecg-heartRate#1":(20,250),"spO2-satO2#1":(50,100),"ecgResp-respRate#1":(4,60)}
splitmap={r["aMRN"]:r["split"] for r in csv.DictReader(open(f"{R}/design/SPLIT_ASSIGNMENT.csv"))}
adms=[(r["aMRN"],r["aCSN"]) for r in csv.DictReader(open(SAMP)) if splitmap.get(r["aMRN"])==SPLIT]
adms=[a for i,a in enumerate(adms) if i%SUB==PHASE]
adms=[a for i,a in enumerate(adms) if i%nstr==stripe]
RS=Resumable(ST,OUT,OUTT,OUTC); st=RS.state            # outputs truncated to the last saved position
t0=time.time(); i=st["i"]; new=RS.is_new(OUT)
fout=RS.files[OUT]; w=csv.writer(fout); ftot=RS.files[OUTT]; wt=csv.writer(ftot); fchg=RS.files[OUTC]; wc=csv.writer(fchg)
if new:
    w.writerow(["aMRN","aCSN","channel","D_s","A","L_s","E_s","S_s","K","lim","over","meanexc","slope10","pre30","nmiss","nchg_before","chg_adm","t_rel_chg"])
    wc.writerow(["aMRN","aCSN","channel","t_change","old","new","direction","magnitude"])
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
        # eligible crossing samples with their excess beyond the contemporaneous limit: the same
        # routine the joint replay scanner uses, so the two agree cell for cell at offset zero
        cross,excv=crossings(vv,sil,silidx,lim_at,dirn)
        rt=sorted(rec.get(base,[]))
        # recorded limit changes on this channel (consecutive rows whose relevant limit differs)
        chg=[]; prev=None
        for (t,lv,hv) in L:
            x=hv if dirn=="high" else lv
            if x is None: continue
            try: xv=float(x)
            except ValueError: continue
            if prev is not None and xv!=prev[1]:
                if xv<=0 or prev[1]<=0:                          # alarm switched off / on (limit recorded as -1)
                    wc.writerow([m,a,base,t,prev[1],xv,("off" if xv<=0 else "on"),""])
                else:
                    chg.append((t,prev[1],xv)); wc.writerow([m,a,base,t,prev[1],xv,("looser" if ((xv>prev[1]) if dirn=="high" else (xv<prev[1])) else "tighter"),abs(xv-prev[1])])
            prev=(t,xv)
        chgt=[c[0] for c in chg]
        vt=[t for t,_ in vv]; vval=[v for _,v in vv]
        def context(s,th):
            """Features of the samples *before* an excursion, which are not crossing samples and
            are read from every plausible measurement: entering slope and pre-excursion level."""
            i0=bisect.bisect_left(vt,s)
            # entering slope: value at first sample of the excursion minus value 10 s before the start
            j=bisect.bisect_right(vt,s-10)-1
            slope=None
            if i0<len(vt) and j>=0 and s-14<=vt[j]<=s-10: slope=(vval[i0]-vval[j])/(s-vt[j])
            # pre-excursion level: mean value over [s-30, s-2], as distance beyond the limit (negative = within)
            k0=bisect.bisect_left(vt,s-30); k1=bisect.bisect_right(vt,s-2)
            pre=None
            if th is not None and k1>k0:
                pv=vval[k0:k1]; pm=sum(pv)/len(pv); pre=(pm-th) if dirn=="high" else (th-pm)
            return slope,pre
        exc4=runs_from(cross,EGAP); ann4=runs_from(rt,EGAP)
        exc30=runs_from(cross,BIG); ann30=runs_from(rt,BIG)
        att,unl4=link(exc4,ann4); _,unl30=link(exc30,ann30)
        # overshoot, mean excess and distinct crossing times of each excursion, over exactly the
        # crossing samples that define it, against the limit in force at each sample; unrounded
        stats=run_stats(cross,excv,exc4)
        for (s,e),runs,(over,meanexc,ncross) in zip(exc4,att,stats):
            D=e-s+2; th=lim_at(s)
            slope,pre=context(s,th)
            nmiss=max((e-s)//2+1-ncross,0)
            nb=bisect.bisect_right(chgt,s); trel=(s-chgt[0]) if chgt else ""
            fx=[("" if th is None else th),over,meanexc,
                ("" if slope is None else round(slope,4)),("" if pre is None else round(pre,3)),nmiss,nb,int(bool(chgt)),trel]
            if not runs: w.writerow([m,a,base,D,0,"","","",0]+fx); continue
            runs.sort(); As=runs[0][0]; Ae=max(r[1] for r in runs); S=sum(r[1]-r[0]+2 for r in runs)
            w.writerow([m,a,base,D,1,As-s,Ae-e,S,len(runs)]+fx)
        wt.writerow([m,a,base,sum(e-s+2 for s,e in exc4),sum(e-s+2 for s,e in ann4),unl4,len(exc4),len(ann4),
                     sum(e-s+2 for s,e in exc30),sum(e-s+2 for s,e in ann30),unl30])
    if i%10==0: save()
save(); fout.close(); ftot.close(); fchg.close()
print(f"{'DONE' if i>=len(adms) else 'PARTIAL'} {SPLIT} stripe={stripe}/{nstr} {i}/{len(adms)} elapsed={time.time()-t0:.0f}s")
