#!/usr/bin/env python3
"""Diagnostics for the transport assumption, the support and the reproducibility of the
policy comparison: representativeness of the analysis subsample, the distribution of
recorded within-admission limit changes, whether duration is a sufficient statistic for
the reward, the two transport conventions tested against each other across recorded
limit changes, the supported share of counterfactual burden, and the comparison
recomputed on a disjoint quarter and with the tuning split's reward.

Environment:
  NBOOT             bootstrap replicates, default 200
  NMIN              minimum excursions before a cell is estimated on its own rather than pooled,
                    default 5, the same variable `analysis/link_analysis.py` reads, so the two
                    programs agree about which cells are supported.
  ALARM_FEATURE_MAP path of a prespecified duration-by-overshoot support map, the file
                    `analysis/link_analysis.py` writes to <work>/policy/FEATURE_MAP.json. When set,
                    the same map is applied here, so one prespecified map governs both programs;
                    when unset, each duration-by-overshoot map is selected at NMIN by the shared
                    `feature_support_map`.
"""
from alarmreplay.paths import WORK as R, COHORT as SAMP
from alarmreplay.linked_reward import feature_support_map
import csv, glob, json, os, sys, collections
import numpy as np
O=f"{R}/policy"; OUT=f"{R}/diagnostics"; os.makedirs(OUT,exist_ok=True)
CH=["ecgResp-numLimit-respRate-low","spO2-numLimit-satO2-low","ecg-numLimit-heartRate-high","ecg-numLimit-heartRate-low"]
NAME={"ecgResp-numLimit-respRate-low":"RR low","spO2-numLimit-satO2-low":"SpO2 low","ecg-numLimit-heartRate-high":"HR high","ecg-numLimit-heartRate-low":"HR low"}
STEP={"ecgResp-numLimit-respRate-low":1.0,"spO2-numLimit-satO2-low":1.0,"ecg-numLimit-heartRate-high":5.0,"ecg-numLimit-heartRate-low":5.0}
import bisect
NB=60; TAILB=[120,180,300,600]; NT=len(TAILB); NC=NB+NT; TAILMID=[150.0,240.0,450.0,900.0]; DS=np.array([2.0*i for i in range(NB)]+TAILMID); OFFS=[-1,0,1,2,3]
NO=6; OBE=[0,1,2,4,8,16]
def cell(d): return int(d//2) if d<120 else NB+bisect.bisect_right(TAILB,d)-1
LA=json.load(open(f"{O}/LINK_ANALYSIS.json"))
rng=np.random.default_rng(20260903); B=int(os.environ.get("NBOOT","200"))
# One threshold and one duration-by-overshoot support map for both programs. NMIN is the minimum
# excursion count before a cell carries its own rate; MAPFILE, when set, is the prespecified map
# analysis/link_analysis.py serializes, applied here unchanged rather than reselected on this sample.
NMIN=int(os.environ.get("NMIN","5"))
MAPFILE=os.environ.get("ALARM_FEATURE_MAP","")
if MAPFILE:
    _m=json.load(open(MAPFILE)); OKMAP={c:np.asarray(_m[c],dtype=bool) for c in CH}
    for c in CH: assert OKMAP[c].shape==(NC,NO), (c,OKMAP[c].shape,(NC,NO))
    MAPSRC=f"prespecified:{os.path.abspath(MAPFILE)}"
else:
    OKMAP=None; MAPSRC="selected on this sample"
def feature_ok(ch,counts):
    """The duration-by-overshoot support map in force for `ch`, on the joint counts `counts`.

    The prespecified map when ALARM_FEATURE_MAP names a file -- intersected with the cells this
    sample actually populates, since a cell the map retains but this sample leaves empty has no rate
    to use -- and otherwise the shared `feature_support_map` at NMIN, which is the rule and the
    threshold `analysis/link_analysis.py` applies.
    """
    if OKMAP is not None:
        return OKMAP[ch]&(np.asarray(counts,dtype=float)>0)
    return feature_support_map(counts,NMIN)
print(f"cell threshold NMIN={NMIN}; duration-by-overshoot map {MAPSRC}")
def _opt(d,k,nd=6):
    """Read a quantity a newer link_analysis.py writes, tolerating a record written before it existed."""
    v=d.get(k)
    return None if v is None else round(v,nd)
def load_rows(tag):
    rows=collections.defaultdict(list)
    for fp in sorted(glob.glob(f"{O}/lk2_rows_{tag}_*.csv")):
        for r in csv.DictReader(open(fp)): rows[r["channel"]].append(r)
    return rows
def load_tot(tag):
    tot=collections.defaultdict(lambda: collections.defaultdict(lambda: np.zeros(8)))
    for fp in sorted(glob.glob(f"{O}/lk2_tot_{tag}_*.csv")):
        for r in csv.DictReader(open(fp)): tot[r["channel"]][r["aMRN"]]+=np.array([float(r[k]) for k in ["exc4","ann4","unl4","nexc4","nann4","exc30","ann30","unl30"]])
    return tot
def load_chg(tag):
    ch=collections.defaultdict(list)
    for fp in sorted(glob.glob(f"{O}/lk2_chg_{tag}_*.csv")):
        for r in csv.DictReader(open(fp)): ch[r["channel"]].append(r)
    return ch
F0=load_rows("FIT4p0"); F2=load_rows("FIT4p2"); TU=load_rows("TUNE2p0")
T0=load_tot("FIT4p0"); T2=load_tot("FIT4p2"); TT=load_tot("TUNE2p0")
C0=load_chg("FIT4p0"); C2=load_chg("FIT4p2"); CT=load_chg("TUNE2p0")
splitmap={r["aMRN"]:r["split"] for r in csv.DictReader(open(f"{R}/design/SPLIT_ASSIGNMENT.csv"))}
samp=[r for r in csv.DictReader(open(SAMP))]
# counterfactual histograms per admission (FIT phase-0 and phase-2 subsamples, gap 4): marginals of the pd3 joint scan
# (60 two-second cells + 4 tail bins with their seconds); the full-FIT descriptive comparison uses the whole-split marginal replay (61 cells)
hist=collections.defaultdict(lambda: collections.defaultdict(lambda: np.zeros((len(OFFS),NC))))
tailsec=collections.defaultdict(lambda: collections.defaultdict(lambda: np.zeros((len(OFFS),NT))))
jhist=collections.defaultdict(lambda: collections.defaultdict(lambda: np.zeros((len(OFFS),NC,NO))))
jsec=collections.defaultdict(lambda: collections.defaultdict(lambda: np.zeros((len(OFFS),NC,NO))))
seen3={}
for fp in sorted(glob.glob(f"{O}/pd3_rows_[0-9].csv"))+sorted(glob.glob(f"{O}/pd3_rows_p2_[0-9].csv")):
    for r in csv.DictReader(open(fp)):
        try:
            k=(r["aMRN"],r["aCSN"],r["channel"],r["offset_step"]); off=int(r["offset_step"])
            if off not in OFFS: continue
            ent=[tuple(int(x) for x in e.split(":")) for e in r["joint"].split(";")]
            if sum(e[2] for e in ent)!=int(r["n_exc"]) or sum(e[3] for e in ent)!=int(r["sum_secs"]): continue
        except Exception: continue
        seen3[k]=(off,ent)
for (m,a,ch,_),(off,ent) in seen3.items():
    j=OFFS.index(off)
    for c,ob,cnt,sec in ent:
        hist[ch][(m,a)][j][c]+=cnt; jhist[ch][(m,a)][j][c][ob]+=cnt; jsec[ch][(m,a)][j][c][ob]+=sec
        if c>=NB: tailsec[ch][(m,a)][j][c-NB]+=sec
hist1=collections.defaultdict(lambda: collections.defaultdict(lambda: np.zeros(NB+1))); secs1=collections.defaultdict(lambda: collections.defaultdict(float)); seen1=set()
for fp in sorted(glob.glob(f"{O}/pd_rows_*.csv")):
    for r in csv.DictReader(open(fp)):
        k=(r["aMRN"],r["aCSN"],r["channel"],r["offset_step"],r["gap_s"])
        if k in seen1 or int(r["gap_s"])!=4 or int(r["offset_step"])!=0: continue
        seen1.add(k)
        hist1[r["channel"]][(r["aMRN"],r["aCSN"])]+=np.array([float(x) for x in r["dwell_hist"].split("|")]+[float(r["n_over120"])])
        secs1[r["channel"]][(r["aMRN"],r["aCSN"])]+=float(r["sum_secs"])
def secs_of(h,ts): return float(h[:NB]@DS[:NB]+ts.sum())
def linked_secs(h,ts,psi,dmean):
    """linked seconds transported cell by cell; within a tail bin the annunciation ratio is held fixed"""
    ps=psi.copy()
    for b in range(NT):
        c=NB+b
        if h[c]>0 and dmean[c]>0: ps[c]=psi[c]*(ts[b]/h[c])/dmean[c]
    return float(h@ps)
res={}
# ---------------------------------------------------------------- 1. representativeness
adm_all=[(r["aMRN"],r["aCSN"],r["tier"]) for r in samp if splitmap.get(r["aMRN"])=="FIT"]
adm_sub={(r["aMRN"],r["aCSN"]) for ch in CH for r in F0[ch]}
adm_sub2={(r["aMRN"],r["aCSN"]) for ch in CH for r in F2[ch]}
def tier_shares(adms):
    c=collections.Counter(t for _,_,t in adms); n=len(adms); return {k:round(v/n,3) for k,v in c.items()}, n, len({m for m,_,_ in adms})
sub_all=[a for a in adm_all if (a[0],a[1]) in adm_sub]; sub2_all=[a for a in adm_all if (a[0],a[1]) in adm_sub2]
rep=[]
ts,n,npat=tier_shares(adm_all); rep.append(["full FIT","admissions",n,"patients",npat]+[f"{k}:{v}" for k,v in sorted(ts.items())])
ts,n,npat=tier_shares(sub_all); rep.append(["every-4th subsample (phase 0)","admissions",n,"patients",npat]+[f"{k}:{v}" for k,v in sorted(ts.items())])
ts,n,npat=tier_shares(sub2_all); rep.append(["every-4th subsample (phase 2)","admissions",n,"patients",npat]+[f"{k}:{v}" for k,v in sorted(ts.items())])
# duration distribution and burden per admission from pd_rows (offset 0), full FIT vs subsample
durrep=[]
for ch in CH:
    def dstats(keys):
        h=np.zeros(NB+1); secs=0.0; nadm=0
        for k in keys:
            if k in hist1[ch]: h+=hist1[ch][k]; secs+=secs1[ch][k]; nadm+=1
        n=h.sum()
        mean=(secs/n) if n else float("nan"); ge60=h[30:].sum()/n if n else float("nan"); tail=h[NB]/n if n else float("nan")
        return dict(n_exc=int(n),mean_D=round(mean,2),share_ge60s=round(ge60,4),share_tail120=round(tail,4),secs_per_adm=round(secs/max(nadm,1),1))
    allk=[(m,a) for m,a,_ in adm_all]; s_all=dstats(allk); s_sub=dstats(list(adm_sub)); s_sub2=dstats(list(adm_sub2))
    durrep.append([NAME[ch],"full FIT"]+[f"{k}={v}" for k,v in s_all.items()]); durrep.append([NAME[ch],"subsample p0"]+[f"{k}={v}" for k,v in s_sub.items()]); durrep.append([NAME[ch],"subsample p2"]+[f"{k}={v}" for k,v in s_sub2.items()])
with open(f"{OUT}/SUBSAMPLE_REPRESENTATIVENESS.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["set","field","value","field","value","tier shares..."]); w.writerows(rep); w.writerow([]); w.writerow(["channel","set","duration and burden summaries (full-FIT pd scan, offset 0, 4-s runs)"]); w.writerows(durrep)
res["representativeness"]={"rows":rep,"durations":durrep}
# ---------------------------------------------------------------- 2. limit changes
lc=[]; res["limit_changes"]={}
for tag,Cc,Rr in (("FIT p0",C0,F0),("FIT p2",C2,F2),("TUNE",CT,TU)):
    for ch in CH:
        adms={(r["aMRN"],r["aCSN"]) for r in Rr[ch]}
        ev=[r for r in Cc[ch] if r["direction"] in ("looser","tighter")]
        offon=[r for r in Cc[ch] if r["direction"] in ("off","on")]
        adm_with=len({(r["aMRN"],r["aCSN"]) for r in ev}); nadm=len(adms)
        mags=np.array([float(r["magnitude"]) for r in ev]); onestep=float(np.mean(np.isclose(mags,STEP[ch]))) if mags.size else float("nan")
        within2=float(np.mean(mags<=2*STEP[ch]+1e-9)) if mags.size else float("nan")
        looser=float(np.mean([r["direction"]=="looser" for r in ev])) if ev else float("nan")
        row=dict(set=tag,channel=NAME[ch],admissions=nadm,admissions_with_change=adm_with,share_with_change=round(adm_with/max(nadm,1),4),
                 n_changes=len(ev),changes_per_admission_with=round(len(ev)/max(adm_with,1),2),share_looser=round(looser,3),
                 median_magnitude=float(np.median(mags)) if mags.size else None,share_one_step=round(onestep,3),share_within_two_steps=round(within2,3),
                 off_on_events=len(offon))
        lc.append(row); res["limit_changes"][f"{tag}|{NAME[ch]}"]=row
with open(f"{OUT}/LIMIT_CHANGE_DISTRIBUTION.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(lc[0].keys())); w.writeheader(); w.writerows(lc)
# ---------------------------------------------------------------- helpers for reward functions
def arrays(rows):
    P=np.array([r["aMRN"] for r in rows]); D=np.array([float(r["D_s"]) for r in rows]); A=np.array([int(r["A"]) for r in rows])
    S=np.array([float(r["S_s"]) if r["A"]=="1" else 0.0 for r in rows]); C=np.array([cell(d) for d in D])
    def fcol(k):
        return np.array([float(r[k]) if r[k] not in ("",None) else np.nan for r in rows])
    Fx={k:fcol(k) for k in ["over","slope10","pre30","lim","nmiss","nchg_before","chg_adm"]}
    ch=rows[0]["channel"] if rows else CH[0]
    Fx["ob"]=np.where(np.isnan(Fx["over"]),0,np.clip(np.searchsorted(OBE,np.nan_to_num(Fx["over"])/STEP[ch],side="right")-1,0,NO-1))
    return P,D,A,S,C,Fx
def cell_means(S,C,mask,pat=None):
    num=np.bincount(C[mask],weights=S[mask],minlength=NC); den=np.bincount(C[mask],minlength=NC)
    return num,den
# ---------------------------------------------------------------- 3. duration sufficiency (FIT p0)
suff=[]; res["duration_sufficiency"]={}
for ch in CH:
    P,D,A,S,C,Fx=arrays(F0[ch]); pats=np.unique(P); pidx={p:i for i,p in enumerate(pats)}; pi=np.array([pidx[p] for p in P])
    num_all=np.bincount(C,weights=S,minlength=NC); den_all=np.bincount(C,minlength=NC)
    a_all=np.where(den_all>0,num_all/np.maximum(den_all,1),0.0); w=den_all/den_all.sum()   # baseline duration weights
    strata={}
    over=Fx["over"]; slope=Fx["slope10"]; pre=Fx["pre30"]; lim=Fx["lim"]; nmiss=Fx["nmiss"]
    def tert(x):
        q=np.nanpercentile(x,[100/3,200/3]); return np.where(np.isnan(x),-1,np.where(x<=q[0],0,np.where(x<=q[1],1,2)))
    strata["overshoot tertile"]=tert(over); strata["entering-slope tertile"]=tert(slope); strata["pre-excursion level tertile"]=tert(pre)
    ul=np.unique(lim[~np.isnan(lim)]); 
    if ul.size>1:
        q=np.nanpercentile(lim,[100/3,200/3]); strata["limit level tertile"]=np.where(np.isnan(lim),-1,np.where(lim<=q[0],0,np.where(lim<=q[1],1,2)))
    strata["signal dropout (missing samples)"]=np.where(np.isnan(nmiss),-1,np.where(nmiss>0,1,0))
    res["duration_sufficiency"][NAME[ch]]={}
    for fname,st in strata.items():
        for lev in sorted(set(st.tolist())-{-1}):
            m=st==lev
            def ratio(idx_pat=None):
                mm=m if idx_pat is None else (m & np.isin(pi,idx_pat))
                num=np.bincount(C[mm],weights=S[mm],minlength=NC); den=np.bincount(C[mm],minlength=NC)
                ok=(den>=NMIN)&(den_all>0)
                a_s=np.where(den>0,num/np.maximum(den,1),0.0)
                # duration-standardised reward: stratum reward function evaluated on the common baseline weights
                return float((w[ok]@a_s[ok])/(w[ok]@a_all[ok])), float(np.sum(w[ok])), float((np.bincount(C[mm],weights=A[mm],minlength=NC)[ok]/np.maximum(den[ok],1))@w[ok]/(w[ok]@(np.bincount(C,weights=A,minlength=NC)/np.maximum(den_all,1))[ok]))
            r0,cov,lr=ratio()
            bs=[]
            for _ in range(B):
                g=rng.integers(0,pats.size,pats.size)
                # resample patients: weight excursions by multiplicity
                cnt=np.bincount(g,minlength=pats.size); wgt=cnt[pi]
                mm=m&(wgt>0)
                num=np.bincount(C[mm],weights=(S*wgt)[mm],minlength=NC); den=np.bincount(C[mm],weights=wgt[mm],minlength=NC)
                numa=np.bincount(C,weights=S*wgt,minlength=NC); dena=np.bincount(C,weights=wgt,minlength=NC)
                ok=(den>=NMIN)&(dena>0); a_s=np.where(den>0,num/np.maximum(den,1e-9),0.0); a_a=np.where(dena>0,numa/np.maximum(dena,1e-9),0.0); ww=dena/dena.sum()
                bs.append((ww[ok]@a_s[ok])/(ww[ok]@a_a[ok]))
            row=dict(channel=NAME[ch],feature=fname,stratum=int(lev),n_excursions=int(m.sum()),reward_ratio_duration_standardised=round(r0,3),
                     CI_lo=round(float(np.percentile(bs,2.5)),3),CI_hi=round(float(np.percentile(bs,97.5)),3),linkage_prob_ratio=round(lr,3),baseline_weight_covered=round(cov,3))
            suff.append(row); res["duration_sufficiency"][NAME[ch]][f"{fname}|{lev}"]=row
with open(f"{OUT}/DURATION_SUFFICIENCY.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(suff[0].keys())); w.writeheader(); w.writerows(suff)
# ---------------------------------------------------------------- 4. transport falsification (FIT p0 + p2)
tr=[]; res["transport"]={}
for ch in CH:
    rows=F0[ch]+F2[ch]; P,D,A,S,C,Fx=arrays(rows); pats=np.unique(P); pidx={p:i for i,p in enumerate(pats)}; pi=np.array([pidx[p] for p in P])
    chg=Fx["chg_adm"]==1; before=chg&(Fx["nchg_before"]==0); after=chg&(Fx["nchg_before"]>=1)
    # direction of the first change per admission
    firstdir={}
    for r in C0[ch]+C2[ch]:
        if r["direction"] in ("looser","tighter"):
            k=(r["aMRN"],r["aCSN"]); 
            if k not in firstdir or float(r["t_change"])<firstdir[k][0]: firstdir[k]=(float(r["t_change"]),r["direction"])
    adkey=[(r["aMRN"],r["aCSN"]) for r in rows]; dirv=np.array([firstdir.get(k,(0,""))[1] for k in adkey])
    def gam(mb,ma,wgt=None):
        wgt=np.ones(len(rows)) if wgt is None else wgt
        nb_=np.bincount(C[mb],weights=(S*wgt)[mb],minlength=NC); db=np.bincount(C[mb],weights=wgt[mb],minlength=NC)
        na_=np.bincount(C[ma],weights=(S*wgt)[ma],minlength=NC); da=np.bincount(C[ma],weights=wgt[ma],minlength=NC)
        ok=(db>=NMIN)&(da>=NMIN); ww=(db+da)[ok]; ab=nb_[ok]/db[ok]; aa=na_[ok]/da[ok]
        return float((ww@aa)/(ww@ab)) if ww.sum()>0 and (ww@ab)>0 else float("nan"), int(ok.sum())
    for label,mb,ma in (("all changes",before,after),("first change looser",before&(dirv=="looser"),after&(dirv=="looser")),("first change tighter",before&(dirv=="tighter"),after&(dirv=="tighter"))):
        g0,ncell=gam(mb,ma); bs=[]
        for _ in range(B):
            g=rng.integers(0,pats.size,pats.size); cnt=np.bincount(g,minlength=pats.size); wgt=cnt[pi].astype(float)
            v,_=gam(mb,ma,wgt); bs.append(v)
        bs=np.array(bs); bs=bs[np.isfinite(bs)]
        row=dict(channel=NAME[ch],comparison=label,admissions_with_change=len({k for k,c in zip(adkey,chg) if c and (label=="all changes" or firstdir.get(k,(0,""))[1]==label.split()[-1])}),
                 n_before=int(mb.sum()),n_after=int(ma.sum()),gamma_after_over_before=round(g0,3),CI_lo=round(float(np.percentile(bs,2.5)),3) if bs.size else None,CI_hi=round(float(np.percentile(bs,97.5)),3) if bs.size else None,cells_compared=ncell)
        tr.append(row); res["transport"][f"{NAME[ch]}|{label}"]=row
with open(f"{OUT}/TRANSPORT_FALSIFICATION_RESULTS.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(tr[0].keys())); w.writeheader(); w.writerows(tr)
# ---------------------------------------------------------------- 4b. which transport convention predicts the after-change reward better?
# observed linked seconds of after-change excursions vs the prediction from the before-change reward function of the same
# admissions under (d) duration-only transport a(c) and (z) duration x relative-overshoot transport a(c,ob) (fallback a(c))
tc=[]; res["transport_comparison"]={}
for ch in CH:
    rows=F0[ch]+F2[ch]; P,D,A,S,C,Fx=arrays(rows); pats=np.unique(P); pidx={p:i for i,p in enumerate(pats)}; pi=np.array([pidx[p] for p in P]); OB=Fx["ob"]
    chg=Fx["chg_adm"]==1; before=chg&(Fx["nchg_before"]==0); after=chg&(Fx["nchg_before"]>=1)
    firstdir={}
    for r in C0[ch]+C2[ch]:
        if r["direction"] in ("looser","tighter"):
            k=(r["aMRN"],r["aCSN"])
            if k not in firstdir or float(r["t_change"])<firstdir[k][0]: firstdir[k]=(float(r["t_change"]),r["direction"])
    adkey=[(r["aMRN"],r["aCSN"]) for r in rows]; dirv=np.array([firstdir.get(k,(0,""))[1] for k in adkey])
    def pred(mb,ma,wgt=None):
        wgt=np.ones(len(rows)) if wgt is None else wgt
        nb1=np.bincount(C[mb],weights=(S*wgt)[mb],minlength=NC); db1=np.bincount(C[mb],weights=wgt[mb],minlength=NC)
        a1=np.where(db1>=NMIN,nb1/np.maximum(db1,1e-9),np.nan)
        nb2=np.zeros((NC,NO)); db2=np.zeros((NC,NO)); np.add.at(nb2,(C[mb],OB[mb]),(S*wgt)[mb]); np.add.at(db2,(C[mb],OB[mb]),wgt[mb])
        a2=np.where(feature_ok(ch,db2),nb2/np.maximum(db2,1e-9),a1[:,None])
        ok=ma&~np.isnan(a1[C])                                            # after-change excursions in cells supported before the change
        obs=float((S*wgt)[ok].sum()); pd_=float((a1[C]*wgt)[ok].sum()); pz=float((a2[C,OB]*wgt)[ok].sum())
        return obs,pd_,pz,int(ok.sum())
    for label,mb,ma in (("all changes",before,after),("first change looser",before&(dirv=="looser"),after&(dirv=="looser")),("first change tighter",before&(dirv=="tighter"),after&(dirv=="tighter"))):
        obs,pd_,pz,nok=pred(mb,ma); bs=[]
        for _ in range(B):
            g=rng.integers(0,pats.size,pats.size); cnt=np.bincount(g,minlength=pats.size); wgt=cnt[pi].astype(float)
            o_,d_,z_,_=pred(mb,ma,wgt); bs.append((o_/d_ if d_>0 else np.nan, o_/z_ if z_>0 else np.nan, z_/d_ if d_>0 else np.nan))
        bs=np.array(bs)
        def ci(j):
            v=bs[:,j]; v=v[np.isfinite(v)]; return (round(float(np.percentile(v,2.5)),3),round(float(np.percentile(v,97.5)),3)) if v.size else (None,None)
        row=dict(channel=NAME[ch],comparison=label,n_after_in_supported_cells=nok,observed_after_linked_s=int(obs),
                 gamma_duration_only=round(obs/pd_,3) if pd_>0 else None,gamma_d_CI_lo=ci(0)[0],gamma_d_CI_hi=ci(0)[1],
                 gamma_feature=round(obs/pz,3) if pz>0 else None,gamma_z_CI_lo=ci(1)[0],gamma_z_CI_hi=ci(1)[1],
                 predicted_ratio_feature_over_duration=round(pz/pd_,3) if pd_>0 else None,ratio_CI_lo=ci(2)[0],ratio_CI_hi=ci(2)[1])
        tc.append(row); res["transport_comparison"][f"{NAME[ch]}|{label}"]=row
with open(f"{OUT}/TRANSPORT_COMPARISON.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(tc[0].keys())); w.writeheader(); w.writerows(tc)
# ---------------------------------------------------------------- 5. support and tail (FIT p0): taken from LINK_ANALYSIS.json (64 cells, pd2 tail bins)
sup=[]; res["support"]={}
for ch in CH:
    la=LA[ch]
    for o in OFFS:
        sp=la["support"][str(o)]; kp=la["kappa"][str(o)]; bo=la["transport_feature"]["by_offset"][str(o)]
        row=dict(channel=NAME[ch],shift=o,replayed_seconds=int(sp["replayed_seconds"]),share_in_cells_with_ge20_baseline=round(sp["share_cells_ge20_baseline"],4),
                 share_unsupported=round(sp["share_unsupported"],4),tail_share_of_seconds=round(sp["share_tail_ge120"],4),share_bin_ge600=round(sp["share_bin_ge600"],4),
                 n_bin_ge600=sp["n_bin_ge600"],baseline_n_bin_ge600=sp["baseline_n_bin_ge600"],
                 # `share_unsupported` above is the DURATION marginal; the estimand integrates over the
                 # joint measure, so these two carry the support condition that matches it, and the two
                 # fallback shares say what the coarse rate costs in seconds and in linked burden.
                 share_unsupported_joint_cells=_opt(sp,"share_unsupported_joint"),
                 share_sparse_joint_cells=_opt(sp,"share_sparse_joint"),
                 share_target_seconds_fallback=_opt(bo,"share_target_seconds_fallback"),
                 share_linked_burden_fallback=_opt(bo,"share_linked_burden_fallback"),
                 kappa_tail_fraction_fixed=(None if o==0 else kp["kappa"]),kappa_tail_seconds_fixed=(None if o==0 else kp["kappa_tail_seconds_fixed"]),
                 kappa_feature_transport=(None if o==0 else bo["kappa_z"]),
                 outside_support=bool(sp["n_bin_ge600"]>3*max(sp["baseline_n_bin_ge600"],1) and sp["share_bin_ge600"]>0.2))
        sup.append(row); res["support"][f"{NAME[ch]}|{o}"]=row
with open(f"{OUT}/DURATION_SUPPORT_AND_TAIL.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(sup[0].keys())); w.writeheader(); w.writerows(sup)
# ---------------------------------------------------------------- 6. reproducibility
rep2=[]; res["reproducibility"]={}
for ch in CH:
    out={}
    for tag,Rr,Tt in (("FIT p0",F0,T0),("FIT p2",F2,T2),("TUNE",TU,TT)):
        P,D,A,S,C,Fx=arrays(Rr[ch]); den=np.bincount(C,minlength=NC); num=np.bincount(C,weights=S,minlength=NC); psi=np.where(den>0,num/np.maximum(den,1),0.0)
        dmean=np.where(den>0,np.bincount(C,weights=D,minlength=NC)/np.maximum(den,1),DS)
        t=np.sum([v for v in Tt[ch].values()],axis=0)
        o=dict(n_excursions=int(D.size),n_patients=int(np.unique(P).size),link_share=round(float(A.mean()),4),rho4=round(float(t[0]/t[1]),3),unlinked_share=round(float(t[2]/t[1]),3),
               lag_median=float(np.median(np.array([float(r["L_s"]) for r in Rr[ch] if r["A"]=="1"]))),E_psi_m=round(float(S.mean()),3),E_D=round(float(D.mean()),3))
        # kappa +1 / -1 applying this set's psi to the FIT counterfactual histograms of the FIT p0 admissions (common durations)
        adms={(r["aMRN"],r["aCSN"]) for r in F0[ch]}; Hs=np.zeros((len(OFFS),NC)); Ts=np.zeros((len(OFFS),NT))
        for k in adms:
            if k in hist[ch]: Hs+=hist[ch][k]; Ts+=tailsec[ch][k]
        def rr(j): return linked_secs(Hs[j],Ts[j],psi,dmean)/secs_of(Hs[j],Ts[j])
        for oo in (1,-1):
            o[f"kappa_{'+' if oo>0 else ''}{oo}_on_FIT_durations"]=round(rr(OFFS.index(0))/rr(OFFS.index(oo)),4)
        # own-admission kappa for FIT sets (phase-2 admissions have no counterfactual scan: only phase 0 carries own values)
        adms2={(r["aMRN"],r["aCSN"]) for r in Rr[ch]}; H2=np.zeros((len(OFFS),NC)); T2_=np.zeros((len(OFFS),NT)); nown=0
        for k in adms2:
            if k in hist[ch]: H2+=hist[ch][k]; T2_+=tailsec[ch][k]; nown+=1
        if nown>0:
            def r2(j): return linked_secs(H2[j],T2_[j],psi,dmean)/secs_of(H2[j],T2_[j])
            o["kappa_+1_own"]=round(r2(OFFS.index(0))/r2(OFFS.index(1)),4)
            o["corrected_reduction_+1_own_pct"]=round(float(100*(1-linked_secs(H2[OFFS.index(1)],T2_[OFFS.index(1)],psi,dmean)/linked_secs(H2[OFFS.index(0)],T2_[OFFS.index(0)],psi,dmean))),2)
            o["replayed_reduction_+1_own_pct"]=round(float(100*(1-secs_of(H2[OFFS.index(1)],T2_[OFFS.index(1)])/secs_of(H2[OFFS.index(0)],T2_[OFFS.index(0)]))),2)
            # feature-level transport on the set's own joint histograms (duration cell x relative-overshoot bin)
            OBs=Fx["ob"]; s2=np.zeros((NC,NO)); n2=np.zeros((NC,NO)); d2=np.zeros((NC,NO))
            np.add.at(s2,(C,OBs),S); np.add.at(n2,(C,OBs),1.0); np.add.at(d2,(C,OBs),D)
            h1=np.where(np.bincount(C,weights=D,minlength=NC)>0,num/np.maximum(np.bincount(C,weights=D,minlength=NC),1e-9),0.0)
            okz=feature_ok(ch,n2)
            a2=np.where(okz,s2/np.maximum(n2,1),psi[:,None]); h2=np.where(okz,s2/np.maximum(d2,1e-9),h1[:,None])
            JH=np.zeros((len(OFFS),NC,NO)); JS_=np.zeros((len(OFFS),NC,NO))
            for k in adms2:
                if k in jhist[ch]: JH+=jhist[ch][k]; JS_+=jsec[ch][k]
            def rz(j): return float((JH[j][:NB]*a2[:NB]).sum()+(JS_[j][NB:]*h2[NB:]).sum())/float(JH[j][:NB].sum(1)@DS[:NB]+JS_[j][NB:].sum())
            o["kappa_z_+1_own"]=round(rz(OFFS.index(0))/rz(OFFS.index(1)),4); o["kappa_z_-1_own"]=round(rz(OFFS.index(0))/rz(OFFS.index(-1)),4)
        out[tag]=o; rep2.append(dict(channel=NAME[ch],set=tag,**o))
    # agreement of h(d) between sets over cells with >=20 in both: max abs difference of psi/d, weighted mean abs diff
    def hcurve(Rr):
        P,D,A,S,C,Fx=arrays(Rr[ch]); den=np.bincount(C,minlength=NC); num=np.bincount(C,weights=S,minlength=NC); return np.where(den>0,num/np.maximum(den,1),0.0)/np.maximum(np.where(den>0,np.bincount(C,weights=D,minlength=NC)/np.maximum(den,1),DS),1e-9), den
    h0,d0=hcurve(F0); h2,d2=hcurve(F2); ht,dt=hcurve(TU)
    for lab,hh,dd in (("FIT p2",h2,d2),("TUNE",ht,dt)):
        ok=(d0>=20)&(dd>=20); w=(d0+dd)[ok]
        out[lab]["h_mean_abs_diff_vs_FITp0"]=round(float(np.sum(w*np.abs(hh[ok]-h0[ok]))/w.sum()),4); out[lab]["h_max_abs_diff_vs_FITp0"]=round(float(np.max(np.abs(hh[ok]-h0[ok]))),4)
        for row in rep2:
            if row["channel"]==NAME[ch] and row["set"]==lab: row["h_mean_abs_diff_vs_FITp0"]=out[lab]["h_mean_abs_diff_vs_FITp0"]; row["h_max_abs_diff_vs_FITp0"]=out[lab]["h_max_abs_diff_vs_FITp0"]
    res["reproducibility"][NAME[ch]]=out
with open(f"{OUT}/REPRODUCIBILITY.csv","w",newline="") as f:
    keys=[]; [keys.append(k) for r in rep2 for k in r if k not in keys]; w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rep2)
json.dump(res,open(f"{OUT}/TRANSPORT_DIAGNOSTICS.json","w"),indent=1)
print("written")
for ch in CH:
    print(NAME[ch], {k:(v["kappa_+1_own"] if "kappa_+1_own" in v else v["kappa_+1_on_FIT_durations"]) for k,v in res["reproducibility"][NAME[ch]].items()}, {k:v.get("h_mean_abs_diff_vs_FITp0") for k,v in res["reproducibility"][NAME[ch]].items()})
    for k,v in res["transport"].items():
        if k.startswith(NAME[ch]): print("   transport",k,v["gamma_after_over_before"],v["CI_lo"],v["CI_hi"],"n",v["n_before"],v["n_after"])
    for k,v in res["support"].items():
        if k.startswith(NAME[ch]) and not k.endswith("|0"): print("   support",k,"ge20",v["share_in_cells_with_ge20_baseline"],"tail",v["tail_share_of_seconds"],"bin600",v["share_bin_ge600"],v["n_bin_ge600"],v["baseline_n_bin_ge600"],"outside",v["outside_support"],"kappa frac/secs/feature",v["kappa_tail_fraction_fixed"],v["kappa_tail_seconds_fixed"],v["kappa_feature_transport"],
                                                                 "joint unsup/sparse",v["share_unsupported_joint_cells"],v["share_sparse_joint_cells"],"fallback secs/burden",v["share_target_seconds_fallback"],v["share_linked_burden_fallback"])
    for k,v in res["transport_comparison"].items():
        if k.startswith(NAME[ch]): print("   transport cmp",k,"gamma_d",v["gamma_duration_only"],(v["gamma_d_CI_lo"],v["gamma_d_CI_hi"]),"gamma_z",v["gamma_feature"],(v["gamma_z_CI_lo"],v["gamma_z_CI_hi"]),"pz/pd",v["predicted_ratio_feature_over_duration"],(v["ratio_CI_lo"],v["ratio_CI_hi"]),"n",v["n_after_in_supported_cells"])
