#!/usr/bin/env python3
"""Linked-reward calibration on the linked scan.

Estimates the conditional linked-annunciation reward a_0(d, z) = E(Y | D = d, Z = z) at the limit in
force -- where Y is total linked annunciation seconds and is an observed zero for every replayed
excursion the device did not annunciate -- and standardizes it over the excursion measure generated
by each candidate limit, under duration-only and duration-by-overshoot transport. Reports the linked
burden contrast with a patient-cluster bootstrap, together with the linkage descriptives, the lag law
by current status, the level identity, the gain-loss split, the support of each counterfactual shift
and the likelihood-ratio order diagnostic.

The estimator algebra lives in `alarmreplay.linked_reward` and is shared with the simulation, so the
simulated and applied estimators are the same code: the reward rates, the standardized burden, the
replayed burden, the burden contrast, kappa, the plateau ratio and the support and fallback
diagnostics are all module functions called from here rather than reimplemented. Cells carry counts
below 120 s, where one cell holds one duration, and replayed seconds in the pooled tail bins;
`tools/test_linked_reward.py` pins down the equivalence of the two carriers.

Two conventions this file is careful about, because both were forked before:

  the plateau      `plateau_ratio` is the single definition of every "at or above 60 s" quantity --
                   a pooled ratio of sums from cell 30 to the end of the map, THE POOLED TAIL BINS
                   INCLUDED. The linkage probability (`q_plateau_share_ge60s`, and `level.q_plateau`
                   inside the level route) and the annunciation ratio (`annunciation_ratio_ge60s`)
                   are both taken on that one slice, so the two quantities the results section
                   compares are on one convention.
  support          `support.share_unsupported` and its neighbours are DURATION cells; the
                   duration-by-overshoot estimand integrates over the joint measure, so
                   `support.share_unsupported_joint` and `support.share_sparse_joint` carry the
                   condition that matches the estimand. The fallback is reported by the burden it
                   carries (`transport_feature.by_offset.*.share_target_seconds_fallback` and
                   `share_linked_burden_fallback`), not only by a count of cells.

Environment:
  NBOOT             bootstrap replicates, default 400
  NMIN              minimum excursions before a duration-by-overshoot cell uses its own reward,
                    default 5. `analysis/transport_diagnostics.py` reads the same variable.
  ALARM_COHORT_SUBSET path of a file naming the admissions to analyse (an aCSN column), with
                    ALARM_COHORT_LABEL naming the arm and ALARM_LINK_TAG suffixing its outputs.
                    Unset in the primary analysis. The restriction in force is recorded per
                    channel under `cohort`, so an arm cannot be mistaken for the primary record.
  ALARM_FEATURE_MAP path of a prespecified duration-by-overshoot support map. When set, the map is
                    loaded and applied unchanged; when unset, it is selected on this sample at NMIN
                    and written to <work>/policy/FEATURE_MAP.json for reuse. Prespecifying the map on
                    an independent split is the primary design; selecting it here is the in-sample
                    sensitivity the manuscript reports. Which of the two happened
                    is recorded per channel in `transport_feature`: `map_source`, `map_file`,
                    `map_prespecified`, `map_cells_retained` and `map_cells_total`.
                    `analysis/transport_diagnostics.py` reads the same file, so one map governs both.
"""
from alarmreplay.paths import WORK as R
import csv, glob, json, os, sys, collections, time
import numpy as np
from alarmreplay.truncation import rev_product_limit
from alarmreplay.cohort import restriction
from alarmreplay.linked_reward import (CellMap, feature_support_map, reward_rates,
                                       standardized_burden, replayed_burden, kappa as lr_kappa,
                                       burden_contrast, support_diagnostics,
                                       joint_support_diagnostics, fallback_diagnostics,
                                       plateau_ratio)
from scipy.stats import kendalltau
O=f"{R}/policy"
# Optional restriction of the analysed admissions (alarmreplay/cohort.py). Unset in the primary
# analysis; the oncology arm of the supplement sets it. ALARM_LINK_TAG suffixes this run's two
# outputs so an arm sits beside the primary record rather than replacing it.
KEEP_ADM, COHORT_RECORD = restriction()
TAG = os.environ.get("ALARM_LINK_TAG", "")
if COHORT_RECORD["restricted"] and not TAG:
    raise SystemExit("ALARM_COHORT_SUBSET is set without ALARM_LINK_TAG: a restricted arm must "
                     "write its own outputs, not overwrite the primary record")
CH=["ecgResp-numLimit-respRate-low","spO2-numLimit-satO2-low","ecg-numLimit-heartRate-high","ecg-numLimit-heartRate-low"]
NAME={"ecgResp-numLimit-respRate-low":"RR low","spO2-numLimit-satO2-low":"SpO2 low","ecg-numLimit-heartRate-high":"HR high","ecg-numLimit-heartRate-low":"HR low"}
# Two quantities from other stages of the same working tree, read rather than carried as constants:
# the held-out naive burden ratio per channel (comparison/A3_NAIVE_VS_DELAY_AWARE.csv, written by
# analysis/adjudicate_test.py) and the median onset lag under the 30-second rule (the onset-lead
# histograms comparison/c1_onset_hist_*.json written by scan/scan_onset_lead.py). Both must exist.
def _held_out_ratios():
    fp=f"{R}/comparison/A3_NAIVE_VS_DELAY_AWARE.csv"
    if not os.path.exists(fp): raise SystemExit(f"{fp} is missing: run analysis/adjudicate_test.py before link_analysis.py")
    out={}
    for r in csv.DictReader(open(fp)): out[r["channel"]]=float(r["naive_burden_ratio"])
    missing=[c for c in CH if c not in out]
    if missing: raise SystemExit(f"held-out ratio missing for {missing}")
    return out
def _raw_rule_medians():
    files=sorted(glob.glob(f"{R}/comparison/c1_onset_hist_*.json"))
    if not files: raise SystemExit(f"{R}/comparison/c1_onset_hist_*.json is missing: run scan/scan_onset_lead.py before link_analysis.py")
    H={}; LO=None; BW=None; lo_tail={}
    for fp in files:
        d=json.load(open(fp)); LO=d["lo"]; BW=d["bin_width_s"]
        if d["admissions_done"]<d["admissions_total"]: raise SystemExit(f"{fp} is incomplete ({d['admissions_done']} of {d['admissions_total']} admissions)")
        for ch,h in d["hist"].items():
            H[ch]=[a+b for a,b in zip(H[ch],h)] if ch in H else list(h); lo_tail[ch]=lo_tail.get(ch,0)+d["lo_tail"][ch]
    out={}
    for ch in CH:
        h=H[ch]; tot=sum(h)+lo_tail[ch]+0; acc=lo_tail[ch]; med=None
        for i,c in enumerate(h):
            acc+=c
            if acc>=tot/2: med=LO+BW*i; break
        if med is None: raise SystemExit(f"no onset-lead median for {ch}")
        out[ch]=med
    return out
RHO_TEST=_held_out_ratios()
RAWMED=_raw_rule_medians()
M=3.0; TAILB=[120,180,300,600]; TAILMID=[150.0,240.0,450.0,900.0]
CMAP=CellMap(width=2.0,n_exact=60,tail_edges=tuple(float(x) for x in TAILB),tail_midpoints=tuple(TAILMID))
NB=CMAP.n_exact; NT=CMAP.n_tail; NC=CMAP.n_cells
DS=CMAP.representative_duration; OFFS=[-1,0,1,2,3]
def cell(d): return int(CMAP.index(d))
rows=collections.defaultdict(list); adm=set()
for fp in sorted(glob.glob(f"{O}/lk_rows_*.csv")):
    for r in csv.DictReader(open(fp)):
        if not KEEP_ADM(r): continue
        rows[r["channel"]].append(r); adm.add((r["aMRN"],r["aCSN"]))
tot=collections.defaultdict(lambda: collections.defaultdict(lambda: np.zeros(8)))
for fp in sorted(glob.glob(f"{O}/lk_tot_*.csv")):
    for r in csv.DictReader(open(fp)):
        if not KEEP_ADM(r): continue
        tot[r["channel"]][r["aMRN"]]+=np.array([float(r[k]) for k in ["exc4","ann4","unl4","nexc4","nann4","exc30","ann30","unl30"]])
hist=collections.defaultdict(lambda: collections.defaultdict(lambda: np.zeros((len(OFFS),NC)))); seen=set()
tsec=collections.defaultdict(lambda: collections.defaultdict(lambda: np.zeros((len(OFFS),NT))))     # tail seconds by bin
for fp in sorted(glob.glob(f"{O}/pd2_rows_*.csv")):
    for r in csv.DictReader(open(fp)):
        k=(r["aMRN"],r["aCSN"],r["channel"],r["offset_step"])
        if k in seen or (r["aMRN"],r["aCSN"]) not in adm: continue
        seen.add(k); off=int(r["offset_step"])
        if off in OFFS:
            hist[r["channel"]][r["aMRN"]][OFFS.index(off)]+=np.array([float(x) for x in r["dwell_hist"].split("|")]+[float(x) for x in r["tail_counts"].split("|")])
            tsec[r["channel"]][r["aMRN"]][OFFS.index(off)]+=np.array([float(x) for x in r["tail_secs"].split("|")])
# baseline overshoot of every linked-scan excursion (lk2 feature scan, same excursions in the same order)
NO=6; OBE=[0,1,2,4,8,16]; STEP={"ecg-numLimit-heartRate-high":5.0,"ecg-numLimit-heartRate-low":5.0,"spO2-numLimit-satO2-low":1.0,"ecgResp-numLimit-respRate-low":1.0}
jn=collections.defaultdict(lambda: collections.defaultdict(lambda: np.zeros((len(OFFS),NC,NO))))
js=collections.defaultdict(lambda: collections.defaultdict(lambda: np.zeros((len(OFFS),NC,NO))))
seen3={}
for fp in sorted(glob.glob(f"{O}/pd3_rows_[0-9].csv")):
    for r in csv.DictReader(open(fp)):
        try:
            k=(r["aMRN"],r["aCSN"],r["channel"],r["offset_step"]); off=int(r["offset_step"])
            if (r["aMRN"],r["aCSN"]) not in adm or off not in OFFS: continue
            ent=[tuple(int(x) for x in e.split(":")) for e in r["joint"].split(";")]
            if sum(e[2] for e in ent)!=int(r["n_exc"]) or sum(e[3] for e in ent)!=int(r["sum_secs"]): continue   # truncated line
        except Exception: continue
        seen3[k]=(r["aMRN"],off,ent)                                   # last complete occurrence wins
for k,(m,off,ent) in seen3.items():
    for c,ob,cnt,sec in ent:
        jn[k[2]][m][OFFS.index(off)][c][ob]+=cnt; js[k[2]][m][OFFS.index(off)][c][ob]+=sec
over_rows=collections.defaultdict(list)
for fp in sorted(glob.glob(f"{O}/lk2_rows_FIT4p0_*.csv")):
    for r in csv.DictReader(open(fp)):
        if KEEP_ADM(r): over_rows[r["channel"]].append(r)
rng=np.random.default_rng(20260902); B=int(os.environ.get("NBOOT","400"))
out={}; FEATMAP={}
for ch in CH:
    t0=time.time()
    Rw=rows[ch]
    if len(Rw)<2:
        raise SystemExit(f"{NAME[ch]}: {len(Rw)} excursions after the cohort restriction "
                         f"({COHORT_RECORD['cohort']}); nothing can be estimated on this channel")
    Rw=rows[ch]; P=np.array([r["aMRN"] for r in Rw]); D=np.array([float(r["D_s"]) for r in Rw]); A=np.array([int(r["A"]) for r in Rw])
    S=np.array([float(r["S_s"]) if r["A"]=="1" else 0.0 for r in Rw]); L=np.array([float(r["L_s"]) if r["A"]=="1" else np.nan for r in Rw])
    E=np.array([float(r["E_s"]) if r["A"]=="1" else np.nan for r in Rw]); K=np.array([int(r["K"]) for r in Rw])
    C=np.array([cell(d) for d in D])
    pats=sorted(set(P)|set(tot[ch])|set(hist[ch])); pidx={p:i for i,p in enumerate(pats)}; pi=np.array([pidx[p] for p in P])
    n=len(pats)
    cellS=np.zeros((n,NC)); cellN=np.zeros((n,NC)); cellD=np.zeros((n,NC)); cellA=np.zeros((n,NC))
    np.add.at(cellS,(pi,C),S); np.add.at(cellN,(pi,C),1.0); np.add.at(cellD,(pi,C),D); np.add.at(cellA,(pi,C),A)
    HH=np.array([hist[ch][p] for p in pats])                       # (n, offs, cells)
    TS=np.array([tsec[ch][p] for p in pats])                       # (n, offs, tail bins)
    TT=np.array([tot[ch][p] for p in pats])                        # (n, 8)
    R2=over_rows[ch]; assert len(R2)==len(Rw), (len(R2),len(Rw))
    assert all(r2["D_s"]==r1["D_s"] and r2["A"]==r1["A"] and r2["S_s"]==r1["S_s"] and r2["aMRN"]==r1["aMRN"] for r1,r2 in zip(Rw,R2)), "lk2 rows not aligned with lk rows"
    OV=np.array([float(r["over"]) if r["over"]!="" else np.nan for r in R2])
    OB=np.where(np.isnan(OV),0,np.clip(np.searchsorted(OBE,np.nan_to_num(OV)/STEP[ch],side="right")-1,0,NO-1))
    cellS2=np.zeros((n,NC,NO)); cellN2=np.zeros((n,NC,NO)); cellD2=np.zeros((n,NC,NO))
    np.add.at(cellS2,(pi,C,OB),S); np.add.at(cellN2,(pi,C,OB),1.0); np.add.at(cellD2,(pi,C,OB),D)
    JN=np.array([jn[ch][p] for p in pats]); JS=np.array([js[ch][p] for p in pats])   # (n, offs, cells, obins)
    NMIN=int(os.environ.get("NMIN","5"))
    # The duration-by-overshoot support map. It is fixed before estimation and held fixed inside every
    # bootstrap replicate, so the estimator is one deterministic smooth function of the patient means.
    # Prespecifying it on an independent split makes that exact; selecting it here makes it exact only in
    # the limit, under stabilization conditions. A map cell that happens to
    # be empty in a replicate falls back to the duration-only reward for that replicate; the event has
    # vanishing probability as n grows and is counted (empty_map_cell_events_in_bootstrap).
    MAPFILE=os.environ.get("ALARM_FEATURE_MAP","")
    if MAPFILE:
        OKMAP=np.asarray(json.load(open(MAPFILE))[ch],dtype=bool); MAPSRC=f"prespecified:{MAPFILE}"
        assert OKMAP.shape==(NC,NO), (OKMAP.shape,(NC,NO))
    else:
        OKMAP=feature_support_map(cellN2.sum(0),NMIN); MAPSRC="selected on this sample"
    FEATMAP[ch]=OKMAP
    # the record of the map actually in force, so a run can be told apart from one that loaded a
    # different file or none at all
    MAPPROV={"map_source":MAPSRC,"map_file":(os.path.abspath(MAPFILE) if MAPFILE else None),
             "map_prespecified":bool(MAPFILE),"map_fixed_before_resampling":True,
             "map_cells_retained":int(OKMAP.sum()),"map_cells_total":int(OKMAP.size),
             "map_share_retained":round(float(OKMAP.sum()/OKMAP.size),4)}
    empty_events=[]
    def functionals_z(idx,okmap=OKMAP):
        s2=cellS2[idx].sum(0); n2=cellN2[idx].sum(0); d2=cellD2[idx].sum(0); s1=cellS[idx].sum(0); n1=cellN[idx].sum(0); d1=cellD[idx].sum(0)
        JNs=JN[idx].sum(0); JSs=JS[idx].sum(0)
        psi1=reward_rates(s1,n1); h1=reward_rates(s1,d1)
        ok=okmap&(n2>0); empty_events.append(int((okmap&(n2==0)).sum()))
        # duration-by-overshoot reward on the same carriers: counts in exact cells, seconds in tail cells,
        # coarsened to the duration-only rate outside the prespecified map
        a2=reward_rates(s2,np.where(ok,n2,0.0),fallback=psi1)          # linked seconds per excursion
        h2=reward_rates(s2,np.where(ok,d2,0.0),fallback=h1)            # annunciation ratio, for the tail bins
        beta_z=np.concatenate([a2[:NB],h2[NB:]])                       # reward per carrier unit, joint cells
        out_={}; jsup={}; jfb={}
        for j,o in enumerate(OFFS):
            rep=replayed_burden(CMAP,JNs[j].sum(1),JSs[j][NB:])
            lin=float((JNs[j][:NB]*a2[:NB]).sum()+(JSs[j][NB:]*h2[NB:]).sum())
            lin1=float((JNs[j][:NB]*psi1[:NB,None]).sum()+(JSs[j][NB:]*h1[NB:,None]).sum())      # duration-only transport on the same joint histogram
            out_[o]=(rep,lin,lin1)
            # support and fallback on the JOINT cells the estimand integrates over, weighted by the
            # target's replayed seconds and by the standardized linked burden rather than counted
            jsup[o]=joint_support_diagnostics(CMAP,n2,JNs[j],JSs[j],nmin=NMIN)
            jfb[o]=fallback_diagnostics(CMAP,ok,beta_z,JNs[j],JSs[j])
        kz={o:lr_kappa(out_[0][1],out_[o][1],out_[0][0],out_[o][0]) for o in OFFS}
        k1={o:lr_kappa(out_[0][2],out_[o][2],out_[0][0],out_[o][0]) for o in OFFS}
        cz={o:burden_contrast(out_[o][1],out_[0][1]) for o in OFFS}
        shob={o:(JNs[j].sum(0)/max(JNs[j].sum(),1)).tolist() for j,o in enumerate(OFFS)}            # excursion share by overshoot bin
        return kz,k1,cz,shob,int((~okmap[:NB]).sum()),{o:out_[o][0] for o in OFFS},jsup,jfb
    kz,k1z,cz,shob,nfb,repz,jsup,jfb=functionals_z(np.arange(n)); empty_events.clear()
    # sensitivity of the point estimate to the coarsening threshold (map re-selected on the full sample at each threshold)
    nmin_sens={str(m):{str(o):round(float(functionals_z(np.arange(n),okmap=(cellN2.sum(0)>=m))[0][o]),4) for o in OFFS} for m in (5,10,20,50)}
    empty_events.clear()
    # likelihood-ratio premise of the replay-direction argument on the replayed duration histograms (baseline vs +1): the ratio of the
    # two histograms should be nonincreasing in duration; reported as the burden-weighted share of adjacent supported
    # cells where it increases (a diagnostic, not a test; the application relies on the ST order of the length-biased laws)
    def lr_check(j):
        h0_=HH.sum(0)[OFFS.index(0)]; h1_=HH.sum(0)[j]; sup=(h0_>=20)&(h1_>=20); idx_=np.flatnonzero(sup[:NB])
        r_=h1_[idx_]/h0_[idx_]; w_=(h0_[idx_]*DS[idx_]); inc=np.diff(r_)>0
        share=float((w_[1:]*inc).sum()/max(w_[1:].sum(),1e-9))
        from scipy.stats import spearmanr
        rho_=float(spearmanr(DS[idx_],r_).statistic) if idx_.size>3 else float("nan")
        return {"share_burden_ratio_increasing":round(share,3),"spearman_ratio_vs_duration":round(rho_,3),"n_cells":int(idx_.size)}
    lr_diag={str(o):lr_check(j) for j,o in enumerate(OFFS) if o!=0}
    def functionals(idx):
        s=cellS[idx].sum(0); nn=cellN[idx].sum(0); dd=cellD[idx].sum(0); aa=cellA[idx].sum(0); h=HH[idx].sum(0); t=TT[idx].sum(0); ts=TS[idx].sum(0)
        psi=reward_rates(s,nn)                                     # linked seconds per excursion by cell
        dmean=reward_rates(dd,nn,fallback=DS)                      # mean duration in cell at the recorded limit
        sh=np.where(nn>0, aa/np.maximum(nn,1), np.nan)             # annunciation probability by cell
        # replayed seconds under each shift: exact (2-s cells are single durations; tail bins carry their own seconds)
        rep_secs={o:replayed_burden(CMAP,h[j],ts[j]) for j,o in enumerate(OFFS)}
        # linked seconds under each shift: psi transported cell by cell; within a tail bin the annunciated
        # FRACTION is held fixed (psi scaled by the bin's mean duration under the shift) -- primary; the
        # alternative holds the linked SECONDS fixed
        # generic carrier: counts in the exact cells, replayed seconds in the pooled tail cells.
        # beta = linked seconds per carrier unit; the standardized burden is beta . R(u) in both.
        carrier_base=np.concatenate([nn[:NB],dd[NB:]])
        beta=reward_rates(s,carrier_base)
        def linked(j,scale):
            if scale:
                return standardized_burden(beta,np.concatenate([h[j][:NB],ts[j]]))
            return float(h[j]@psi)                                   # tail linked SECONDS held fixed
        L1={o:linked(j,True) for j,o in enumerate(OFFS)}; L0={o:linked(j,False) for j,o in enumerate(OFFS)}
        kap={o:lr_kappa(L1[0],L1[o],rep_secs[0],rep_secs[o]) for o in OFFS}
        kap_const={o:lr_kappa(L0[0],L0[o],rep_secs[0],rep_secs[o]) for o in OFFS}
        red={o:burden_contrast(rep_secs[o],rep_secs[0]) for o in OFFS}                # replayed reduction
        corr={o:burden_contrast(L1[o],L1[0]) for o in OFFS}                           # corrected (linked seconds)
        lvl={"rho4":t[0]/t[1],"rho30":t[5]/t[6],"unl4_share":t[2]/t[1],"unl30_share":t[7]/t[6],"rho4_linked":t[0]/(t[1]-t[2])}
        # level route under the same convention: persistence rule, q = plateau of the linkage probability,
        # the one tail-inclusive pooled ratio of `plateau_ratio`
        qpl=plateau_ratio(aa,nn)
        target=1.0/(qpl*lvl["rho4"])
        h0=h[OFFS.index(0)]
        def dm_shift(j):
            d=dmean.copy()
            for b in range(NT):
                c=NB+b
                if h[j][c]>0: d[c]=ts[j][b]/h[j][c]
            return d
        def lvl_burden(j,th): return float(h[j]@np.maximum(dm_shift(j)-th,0))   # persistence-rule burden under shift j
        def rth(th,j=OFFS.index(0)): d=dm_shift(j); return float((h[j]@np.maximum(d-th,0))/rep_secs[OFFS[j]])
        if target<1.0 and rth(0.0)>=target:
            lo_,hi_=0.0,200.0
            for _ in range(60):
                mid=0.5*(lo_+hi_)
                if rth(mid)>target: lo_=mid
                else: hi_=mid
            th=0.5*(lo_+hi_)
            j0=OFFS.index(0)
            kl={o:lr_kappa(lvl_burden(j0,th),lvl_burden(j,th),rep_secs[0],rep_secs[o]) for j,o in enumerate(OFFS)}
            # the contrast keeps the baseline cell-mean durations in its denominator (dmean, not the
            # shift-0 counterfactual means dm_shift used by kappa)
            cl_base=float(h0@np.maximum(dmean-th,0))
            cl={o:burden_contrast(lvl_burden(j,th),cl_base) for j,o in enumerate(OFFS)}
        else:
            th=float("nan"); kl={o:float("nan") for o in OFFS}; cl={o:float("nan") for o in OFFS}
        lvl["q_plateau"]=qpl; lvl["theta_star_4s"]=th
        functionals.rep_secs=rep_secs
        return psi,dmean,sh,kap,red,corr,lvl,kl,cl,kap_const
    psi,dmean,sh,kap,red,corr,lvl,kl,cl,kap_const=functionals(np.arange(n)); rep_secs_pt=dict(functionals.rep_secs)
    hboot=[]
    boot=collections.defaultdict(list); empty_events.clear()
    for _ in range(B):
        idx=rng.integers(0,n,n); psb,dmb,_,kb,_,cb,lb,klb,clb,kcb=functionals(idx)
        for o in OFFS: boot[f"kappa{o}"].append(kb[o]); boot[f"corr{o}"].append(cb[o]); boot[f"kappaL{o}"].append(klb[o]); boot[f"corrL{o}"].append(clb[o]); boot[f"kappaC{o}"].append(kcb[o])
        for k,v in lb.items(): boot[k].append(v)
        hboot.append(psb/np.maximum(dmb,1e-9))
        kzb,k1b,czb,_,_,_,_,_=functionals_z(idx)
        for o in OFFS: boot[f"kappaZ{o}"].append(kzb[o]); boot[f"dZ{o}"].append(kzb[o]-kb[o]); boot[f"corrZ{o}"].append(czb[o])
    ci={k:([float(np.nanpercentile(v,2.5)),float(np.nanpercentile(v,97.5))] if np.isfinite(v).sum()>=20 else [float("nan"),float("nan")]) for k,v in boot.items()}
    hboot=np.array(hboot); h_ci_lo=np.nanpercentile(hboot,2.5,axis=0); h_ci_hi=np.nanpercentile(hboot,97.5,axis=0)
    # Support is reported on two different cell systems and they must not be confused. `support_diagnostics`
    # works on the DURATION marginal; the duration-by-overshoot estimand integrates over the JOINT measure,
    # so `joint_support_diagnostics` (computed inside functionals_z on the pd3 joint histograms the estimator
    # actually standardizes) carries the condition that matches the estimand. Both go into the same block.
    NBASE_DURATION=cellN.sum(0)                    # baseline excursions per duration cell
    def duration_support_block(j,o):
        sd=support_diagnostics(CMAP,NBASE_DURATION,HH.sum(0)[j],TS.sum(0)[j],min_for_precision=20)
        return {"support_basis":"unsuffixed keys are duration cells; *_joint keys are the joint (duration x overshoot) cells",
                "replayed_seconds":sd["replayed_seconds"],
                "share_cells_ge20_baseline":sd["share_supported_precise"],   # duration cells
                "share_unsupported":sd["share_unsupported"],                 # duration cells
                "share_tail_ge120":sd["share_pooled_tail"],
                "share_bin_ge600":sd["share_last_pooled_cell"],
                "n_bin_ge600":int(sd["n_last_pooled_cell"]),
                "baseline_n_bin_ge600":int(sd["baseline_n_last_pooled_cell"]),
                "replayed_seconds_joint":jsup[o]["replayed_seconds_joint"],
                "share_unsupported_joint":jsup[o]["share_unsupported_joint"],
                "share_sparse_joint":jsup[o]["share_sparse_joint"]}
    # ONE estimand for the high-duration plateau, computed by `plateau_ratio` at every site: the pooled
    # ratio of sums over all cells at or above 60 s -- the four pooled tail cells included -- weighting
    # each cell by the mass it holds, the same quantity `functionals` records as `level.q_plateau`. The
    # annunciation ratio is taken on the same slice, so the two quantities the results section compares
    # are on one convention.
    q_plateau=plateau_ratio(cellA.sum(0),cellN.sum(0))                   # linkage probability >= 60 s
    ann_ratio=plateau_ratio(cellS.sum(0),cellD.sum(0))                   # annunciation ratio  >= 60 s
    q_plateau_exact_only=plateau_ratio(cellA.sum(0)[:NB],cellN.sum(0)[:NB])   # the superseded tail-exclusive slice
    ann_ratio_exact_only=plateau_ratio(cellS.sum(0)[:NB],cellD.sum(0)[:NB])
    # lag law: current status (the annunciation probability by cell, normalized by the plateau) vs product-limit
    Lp=L[A==1]; Dp=D[A==1]; Ep=E[A==1]
    Lj=Lp+rng.uniform(-0.5,0.5,Lp.size); u,F=rev_product_limit(Lj, Dp-M+0.5)
    def Fpl(x):
        k=np.clip(np.searchsorted(u,x,side="right")-1,0,len(u)-1); return float(np.where(x<u[0],0.0,F[k]))
    xs=[2,5,8,12,17,22,27,37,57,87]
    lag_cmp={str(x):{"current_status_qF":float(sh[cell(x+M)]) if not np.isnan(sh[cell(x+M)]) else None,
                     "current_status_F": (float(sh[cell(x+M)])/q_plateau if not np.isnan(sh[cell(x+M)]) else None),
                     "product_limit_F":Fpl(x+0.5)} for x in xs}
    # persistence-form check on the same admissions: E[D]/(q E[(D-theta)+]) vs rho4_linked
    ED=float(D.mean()); 
    def rform(th): return float(np.maximum(D-th,0).mean()/ED)
    from scipy.optimize import brentq
    r_m=float(S.mean()/ED)                                          # E[psi_m]/E[D]
    # The fixed offset reproducing the observed linked seconds exists only when the linked-second
    # ratio lies inside the range the offset can produce. It need not, on a small subsample or
    # where annunciations outlast their excursions; that is a finding about the persistence form,
    # not an error, so it is recorded as absent and nothing downstream uses it.
    THBR=(-60.0,200.0)
    try:
        theta_m=float(brentq(lambda th: rform(th)-r_m, *THBR))      # fixed offset reproducing the linked seconds
    except ValueError:
        theta_m=None
    # dichotomy: psi_m(d)/d on the grid, trough, burden share below, G-L split under u0 vs u1 length-biased
    grid=dmean[1:]; hm=psi[1:]/np.maximum(grid,1e-9)
    ok=cellN.sum(0)[1:]>=20
    tr=int(np.argmin(np.where(ok,hm,np.inf))); d0=float(grid[tr])
    w0=HH.sum(0)[OFFS.index(0)][1:]*grid; w0=w0/w0.sum(); burden_below=float(w0[:tr].sum())
    def GL(j):
        w1=HH.sum(0)[j][1:]*grid; w1=w1/w1.sum()
        hup=np.maximum(hm-hm[tr],0)*(np.arange(len(hm))>=tr); hdn=np.maximum(hm-hm[tr],0)*(np.arange(len(hm))<tr)
        return float(w0@hup-w1@hup), float(w1@hdn-w0@hdn)
    G,Lloss=GL(OFFS.index(1))
    # lag-extension association within exact duration (linked excursions)
    order=np.argsort(Dp,kind="stable"); s_=Dp[order]; b_=np.flatnonzero(np.diff(s_))+1; st=np.concatenate([[0],b_]); en=np.concatenate([b_,[s_.size]])
    num=den=0.0
    for a_,c_ in zip(st,en):
        if c_-a_<3: continue
        ii=order[a_:c_]; tt=kendalltau(Lj[ii],Ep[ii]).statistic
        if np.isfinite(tt): ww=(c_-a_)*(c_-a_-1)/2; num+=tt*ww; den+=ww
    tauLE=num/max(den,1)
    out[ch]={"label":NAME[ch],"cohort":COHORT_RECORD,"n_excursions":int(D.size),"n_linked":int(A.sum()),"n_patients":n,"n_admissions":len({(r['aMRN'],r['aCSN']) for r in Rw}),
             "link_share":float(A.mean()),"multi_run_share_among_linked":float((K[A==1]>1).mean()),"runs_per_linked_excursion":float(K[A==1].mean()),
             "q_plateau_share_ge60s":q_plateau,"annunciation_ratio_ge60s":ann_ratio,
             "plateau_convention":{"from_index":30,"from_duration_s":60.0,"pooled_tail_bins_included":True,
                                   "q_plateau_exact_cells_only":q_plateau_exact_only,
                                   "annunciation_ratio_exact_cells_only":ann_ratio_exact_only},
             "lag_quantiles_linked":{str(qq):float(np.percentile(Lp,qq)) for qq in [50,90,95,99]},
             "E_mean_by_duration":{"<8s":float(Ep[Dp<8].mean()),"8-30s":float(Ep[(Dp>=8)&(Dp<30)].mean()),"30-60s":float(Ep[(Dp>=30)&(Dp<60)].mean()),">=60s":float(Ep[Dp>=60].mean())},
             "share_negative_offset_linked":float(((Lp-Ep)<0).mean()),
             "tau_L_E_within_exact_D":round(float(tauLE),4),
             "lag_law_at_x":lag_cmp,
             "level":{**{k:round(float(v),4) for k,v in lvl.items()},"CI":{k:[round(x,4) for x in ci[k]] for k in lvl},"rho_measured_TEST_frozen":RHO_TEST[ch]},
             "persistence_form":{"E_D_s":round(ED,3),"E_psi_m_s":round(float(S.mean()),3),"r_m":round(r_m,4),"theta_m_s":(round(theta_m,2) if theta_m is not None else None),
                                 "theta_m_note":(None if theta_m is not None else
                                                 f"no fixed offset in [{THBR[0]:.0f}, {THBR[1]:.0f}] s reproduces the observed linked seconds"),
                                 "rho_linked_identity":round(1/r_m,4),"theta_rawmedian":RAWMED[ch],"r_rawmedian":round(rform(RAWMED[ch]),4),
                                 "rho_pred_rawmedian_q_plateau":round(1/(q_plateau*rform(RAWMED[ch])),4)},
             "kappa":{str(o):{"kappa":round(float(kap[o]),4),"CI":[round(x,4) for x in ci[f"kappa{o}"]],"kappa_tail_seconds_fixed":round(float(kap_const[o]),4),"replayed_reduction_pct":round(float(red[o]),2),
                              "corrected_reduction_pct":round(float(corr[o]),2),"corrected_CI":[round(x,2) for x in ci[f"corr{o}"]],
                              "kappa_level":(None if np.isnan(kl[o]) else round(float(kl[o]),4)),"kappa_level_CI":[None if np.isnan(x) else round(x,4) for x in ci[f"kappaL{o}"]],
                              "corrected_level_pct":(None if np.isnan(cl[o]) else round(float(cl[o]),2)),"corrected_level_CI":[None if np.isnan(x) else round(x,2) for x in ci[f"corrL{o}"]]} for o in OFFS},
             "lr_order_diagnostic":lr_diag,
             "transport_feature":{"overshoot_bin_edges_in_steps":OBE,"step_units":STEP[ch],"n_bulk_cells_fallback":nfb,"nmin":NMIN,
                                  **MAPPROV,"empty_map_cell_events_in_bootstrap":{"total":int(sum(empty_events)),"replicates_with_any":int(sum(1 for e in empty_events if e>0)),"B":B},
                                  "kappa_z_by_nmin":nmin_sens,
                                  "replayed_seconds_pd3_equals_pd2":{str(o):bool(abs(repz[o]-rep_secs_pt[o])<1e-6) for o in OFFS},
                                  "kappa_duration_only_on_joint":{str(o):round(float(k1z[o]),4) for o in OFFS},
                                  # the fallback in the units that say what it does to the estimate: the
                                  # share of the candidate's replayed seconds routed through the coarse
                                  # duration-only rate, and the share of the standardized linked burden
                                  # those cells contribute. A count of cells says neither.
                                  "by_offset":{str(o):{"kappa_z":round(float(kz[o]),4),"CI":[round(x,4) for x in ci[f"kappaZ{o}"]],
                                                       "diff_vs_duration_only":round(float(kz[o]-kap[o]),4),"diff_CI":[round(x,4) for x in ci[f"dZ{o}"]],
                                                       "corrected_reduction_z_pct":round(float(cz[o]),2),"corrected_z_CI":[round(x,2) for x in ci[f"corrZ{o}"]],
                                                       "share_target_seconds_fallback":round(jfb[o]["share_target_seconds_fallback"],6),
                                                       "share_linked_burden_fallback":round(jfb[o]["share_linked_burden_fallback"],6),
                                                       "n_joint_cells_fallback":jfb[o]["n_cells_fallback"],
                                                       "excursion_share_by_overshoot_bin":[round(x,4) for x in shob[o]]} for o in OFFS},
                                  "baseline_reward_by_overshoot_bin":{"cells":["<10s","10-20s","20-40s","40-120s",">=120s"],
                                      "h":[[(lambda m: (None if m.sum()<30 else round(float(S[m].sum()/D[m].sum()),3)))((C>=lo)&(C<hi)&(OB==b)) for b in range(NO)] for lo,hi in [(0,5),(5,10),(10,20),(20,NB),(NB,NC)]],
                                      "n":[[int(((C>=lo)&(C<hi)&(OB==b)).sum()) for b in range(NO)] for lo,hi in [(0,5),(5,10),(10,20),(20,NB),(NB,NC)]]}},
             "dichotomy":{"trough_s":d0,"burden_share_below_trough":round(burden_below,4),"G_plus1":round(G,5),"L_plus1":round(Lloss,5)},
             "psi_m_by_cell":{"d_mean_s":np.round(dmean,2).tolist(),"psi_m_s":np.round(psi,3).tolist(),"ann_prob":[None if np.isnan(x) else round(float(x),4) for x in sh],"n":cellN.sum(0).astype(int).tolist(),
                              "h_CI_lo":np.round(h_ci_lo,4).tolist(),"h_CI_hi":np.round(h_ci_hi,4).tolist(),"tail_bins_s":TAILB},
             "hist_by_offset":{str(o):HH.sum(0)[j].tolist() for j,o in enumerate(OFFS)},
             "tail_secs_by_offset":{str(o):TS.sum(0)[j].tolist() for j,o in enumerate(OFFS)},
             "support":{str(o):duration_support_block(j,o) for j,o in enumerate(OFFS)},
             "lag_curves":{"x_s":[float(x) for x in np.arange(-5,61)],
                           "observed_ecdf":[float(np.mean(Lp<=x)) for x in np.arange(-5,61)],
                           "product_limit":[Fpl(x+0.5) for x in np.arange(-5,61)],
                           "current_status":[(None if (x+M)<0 or np.isnan(sh[cell(x+M)]) else float(sh[cell(x+M)]/q_plateau)) for x in np.arange(-5,61)]},
             "n_bootstrap":B}
    print(f"{NAME[ch]:8s} exc={D.size} linked={A.sum()} ({A.mean():.3f}) multi-run={out[ch]['multi_run_share_among_linked']:.3f} q_plateau={q_plateau:.4f} annratio>=60s={ann_ratio:.4f} "
          f"rho4={lvl['rho4']:.3f} rho30={lvl['rho30']:.3f} unl4={lvl['unl4_share']:.3f} rho4_linked={lvl['rho4_linked']:.3f} theta_m={(f'{theta_m:.1f}' if theta_m is not None else 'none')} theta*4s={lvl['theta_star_4s']:.1f} kL+1={kl[1]:.4f} "
          f"kappa+1={kap[1]:.4f} {ci['kappa1']} kappa-1={kap[-1]:.4f} trough={d0:.0f} below={burden_below:.3f} G={G:.4f} L={Lloss:.4f} tauLE={tauLE:+.3f} [{time.time()-t0:.0f}s]")
    print("    feature transport kappa_z:", {o:(round(kz[o],4),[round(x,4) for x in ci[f'kappaZ{o}']]) for o in OFFS}, "fallback cells", nfb, "rep equal", out[ch]["transport_feature"]["replayed_seconds_pd3_equals_pd2"])
    print("    kappa duration-only on joint:", {o:round(k1z[o],4) for o in OFFS}, " share ob0 by offset:", {o:round(shob[o][0],3) for o in OFFS})
    print("    kappa_z by NMIN:", {m:v["1"] for m,v in nmin_sens.items()}, " empty-map-cell events:", out[ch]["transport_feature"]["empty_map_cell_events_in_bootstrap"], " LR diag +1:", lr_diag["1"])
    print("    plateau (tail-inclusive, >=60 s): q =", round(q_plateau,4), " annunciation ratio =", round(ann_ratio,4),
          " [exact cells only:", round(q_plateau_exact_only,4), "/", round(ann_ratio_exact_only,4), "]")
    print("    joint support % (unsupported/sparse<NMIN):", {o:(round(100*jsup[o]["share_unsupported_joint"],3),round(100*jsup[o]["share_sparse_joint"],3)) for o in OFFS})
    print("    fallback burden % (target secs/linked burden):", {o:(round(100*jfb[o]["share_target_seconds_fallback"],3),round(100*jfb[o]["share_linked_burden_fallback"],3)) for o in OFFS},
          " map:", MAPSRC, f"({int(OKMAP.sum())}/{int(OKMAP.size)} joint cells keep their own reward)")
    print("    lag law x:", {x:(round(v['current_status_F'],3) if v['current_status_F'] else None, round(v['product_limit_F'],3)) for x,v in lag_cmp.items()})
    print("    E mean by D:", {k:round(v,1) for k,v in out[ch]['E_mean_by_duration'].items()}, " lag q:", {k:round(v,1) for k,v in out[ch]['lag_quantiles_linked'].items()})
json.dump(out,open(f"{O}/LINK_ANALYSIS{TAG}.json","w"),indent=1)
# Serialize the duration-by-overshoot support map so a later run can apply it unchanged, and so a run on
# an independent split can produce one to prespecify: ALARM_FEATURE_MAP=<this file> python3 analysis/link_analysis.py
# The map is the only thing a later prespecified run takes from this one, so the file is read back and
# checked cell by cell against the array in force here: if the round trip is exact, that run evaluates
# the same estimator on the same map. tools/test_linked_reward.py exercises the same round trip on
# synthetic maps, and the assertion here covers the real ones.
MAPPATH=f"{O}/FEATURE_MAP{TAG}.json"
json.dump({ch:FEATMAP[ch].tolist() for ch in CH},open(MAPPATH,"w"),indent=0)
_rt=json.load(open(MAPPATH))
for ch in CH:
    back=np.asarray(_rt[ch],dtype=bool)
    assert back.shape==FEATMAP[ch].shape, (ch,back.shape,FEATMAP[ch].shape)
    assert np.array_equal(back,FEATMAP[ch]), f"FEATURE_MAP.json does not round-trip on {ch}"
print(f"wrote {os.path.basename(MAPPATH)} (", {ch:int(FEATMAP[ch].sum()) for ch in CH}, "cells with their own reward );",
      "re-read and checked on all", len(CH), "channels")
