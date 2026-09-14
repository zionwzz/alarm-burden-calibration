#!/usr/bin/env python3
"""Manuscript tables for the linkage, the one-step policy comparison under the two
transport scenarios, the full grid of shifts, the level and the diagnostics.
"""
from alarmreplay.paths import WORK as R, TABLES, ensure
ensure(TABLES)   # the generated tables' directory, created on demand
import json, os, re
import numpy as np
O,P=f"{R}/policy",TABLES
def emit(name,text):
    """Write one generated table, refusing to leave a %%TOKEN%% placeholder behind."""
    left=set(re.findall(r"%%[A-Z]+%%",text))
    if left: raise SystemExit(f"{name}: unsubstituted placeholder(s) {sorted(left)}")
    open(f"{P}/{name}","w").write(text)
D=json.load(open(f"{O}/LINK_ANALYSIS.json"))
CH=["ecgResp-numLimit-respRate-low","spO2-numLimit-satO2-low","ecg-numLimit-heartRate-high","ecg-numLimit-heartRate-low"]
TEX={"RR low":"RR low","SpO2 low":r"\SpO{} low","HR high":"HR high","HR low":"HR low"}
LONG={"RR low":"respiration rate low","SpO2 low":r"\SpO{} low","HR high":"heart rate high","HR low":"heart rate low"}
# The median onset lag under the 30-second rule is the value link_analysis.py records per channel
# (persistence_form.theta_rawmedian, taken from the onset-lead histograms); the held-out naive
# interval is read from the adjudication's own output. Neither is carried here as a constant.
FROZEN_MEDIAN={D[c]["label"]:D[c]["persistence_form"]["theta_rawmedian"] for c in CH}
import csv as _csv
_a3=f"{R}/comparison/A3_NAIVE_VS_DELAY_AWARE.csv"
if not os.path.exists(_a3): raise SystemExit(f"{_a3} is missing: run analysis/adjudicate_test.py")
TEST_CI={}
for _r in _csv.DictReader(open(_a3)):
    _lo,_hi=(float(x) for x in _r["naive_CI"].strip("[]").split(","))
    TEST_CI[D[_r["channel"]]["label"]]=(_lo,_hi)
    if abs(float(_r["naive_burden_ratio"])-D[_r["channel"]]["level"]["rho_measured_TEST_frozen"])>0.005:
        raise SystemExit(f"held-out ratio in the analysis record ({D[_r['channel']]['level']['rho_measured_TEST_frozen']}) "
                         f"disagrees with the adjudication ({_r['naive_burden_ratio']}) for {_r['channel']}")
UNIT={"RR low":"1 breath/min","SpO2 low":r"1\%","HR high":"5 bpm","HR low":"5 bpm"}
summ={}
# ---------------- Table: linkage
rows=[]
for c in CH:
    v=D[c]; L=v["label"]; lq=v["lag_quantiles_linked"]
    rows.append(f"{TEX[L]} & {v['n_excursions']:,} & {100*v['link_share']:.1f}\\% & {100*v['multi_run_share_among_linked']:.1f}\\% & "
                f"{lq['50']:.0f} / {lq['90']:.0f} / {lq['99']:.0f} & {FROZEN_MEDIAN[L]} & {v['E_mean_by_duration']['8-30s']:.1f} & "
                f"{100*v['share_negative_offset_linked']:.1f}\\% & {100*v['level']['unl4_share']:.1f}\\% & {v['q_plateau_share_ge60s']:.2f} \\\\".replace(",","\\,"))
tab=r"""\begin{table}[htbp]\centering\footnotesize
\setlength{\tabcolsep}{3.4pt}
\caption{Linkage of replayed excursions to recorded annunciations on the fitting-split subsample
($1\,543$ admissions), under the theory's convention: excursions and annunciation runs merged at 4\,s,
silenced and inactivated samples excluded from both sides, every annunciation run attached to the excursion
it overlaps most. `Linked' is the share of excursions with at least one attached run --- the linkage
probability averaged over durations; `multi-run' the share of linked excursions with more than one run.
Lag quantiles are onset lags of the earliest attached run; the next column is the median under the
matching rule fixed in advance (30-s runs, one-to-one matching). $\bar E$ is the mean
extension for excursions of 8--30\,s; $\Pr(\Theta<0)$ the share of linked excursions whose alarm
outlasts the excursion by more than it lagged; `unlinked' the share of recorded annunciation seconds
attached to no excursion; $\hat q$ the linkage probability among excursions of at least 60\,s.}
\label{tab:link}
\begin{tabular}{lrrrrrrrrr}
\toprule
 & & & & \multicolumn{2}{c}{Onset lag (s)} & & & & \\
\cmidrule(lr){5-6}
Channel & Excursions & Linked & Multi-run & q50 / q90 / q99 & fixed-rule q50 & $\bar E$ (s) & $\Pr(\Theta<0)$ & Unlinked & $\hat q$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\tightnote{Silenced time is excluded on both sides, so an alarm silenced part-way through an excursion
ends, for this accounting, where the silence begins; on excursions of a minute or more the mean extension
is negative on every channel ($-4$ to $-40$\,s), which is annunciation that stops before the excursion does.}
\end{table}
"""
emit("TABLE_LINKAGE.tex",tab)
# ---------------- Table: the one-step comparison (main) and the full shift grid (supplement)
DG=json.load(open(f"{R}/diagnostics/TRANSPORT_DIAGNOSTICS.json"))
def pc(x): return f"${x:.1f}$\\%"
def cif(ci,nd=3): return f"({ci[0]:.{nd}f}--{ci[1]:.{nd}f})"
def outside(v,o): return DG["support"][f"{v['label']}|{o}"]["outside_support"]
def rowfor(v,o,first,full):
    k=v["kappa"][o]; z=v["transport_feature"]["by_offset"][o]; L=v["label"]
    lab={"-1":"$-1%s$","0":"$0%s$","1":"$+1%s$","2":"$+2%s$","3":"$+3%s$"}[o]%(r"^{\dagger}" if outside(v,o) else "")
    lev="---" if k["kappa_level"] is None else f"{k['kappa_level']:.3f} {cif(k['kappa_level_CI'])}"
    ci=k["corrected_CI"]; cz=z["corrected_z_CI"]
    def red(x,ci):
        if x>=0: return f"{x:.1f} ({ci[0]:.1f}--{ci[1]:.1f})"
        return f"$-${abs(x):.1f} ($-${abs(ci[0]):.1f} to $-${abs(ci[1]):.1f})"
    def red1(x): return f"{x:.1f}" if x>=0 else f"$-${abs(x):.1f}"
    cells=[TEX[L] if first else "",lab,pc(k["replayed_reduction_pct"]),f"{k['kappa']:.3f} {cif(k['CI'])}",red1(k["corrected_reduction_pct"]),
           f"{z['kappa_z']:.3f} {cif(z['CI'])}",red1(z["corrected_reduction_z_pct"]),f"{100*z['share_linked_burden_fallback']:.1f}",f"{k['kappa_tail_seconds_fixed']:.3f}",lev]
    return " & ".join(cells)+r" \\"
rows=[]; rows_full=[]
for c in CH:
    v=D[c]; first=True
    for o in ["-1","1"]:
        if outside(v,o): continue                      # the pooled-tail-sensitive contrast is reported in the supplement only
        rows.append(rowfor(v,o,first,False)); first=False
    rows.append(r"\addlinespace"); first=True
    for o in ["-1","1","2","3"]:
        rows_full.append(rowfor(v,o,first,True)); first=False
    rows_full.append(r"\addlinespace")
rows=rows[:-1]; rows_full=rows_full[:-1]
head=r"""\begin{tabular}{@{}llrlrlrrrl@{}}
\toprule
 & & & \multicolumn{2}{c}{Duration transport} & \multicolumn{3}{c}{Duration $\times$ overshoot} & Tail & Level route \\
\cmidrule(lr){4-5}\cmidrule(lr){6-8}\cmidrule(lr){9-9}\cmidrule(lr){10-10}
Channel & Shift & Replayed & $\hat\kappa$ (95\% CI) & Corr.\ \% & $\hat\kappa_z$ (95\% CI) & Corr.\ \% & Coars.\ \% & $\hat\kappa$ & $\hat\kappa$ (95\% CI) \\
\midrule
"""
# The map sensitivity the note states is measured, not carried: the in-sample arm's LINK_ANALYSIS.json
# (analysis/link_analysis.py run without ALARM_FEATURE_MAP, in a working tree of its own) is read from
# ALARM_INSAMPLE_LINK_ANALYSIS, default <work>/policy/LINK_ANALYSIS_insample.json, and the largest movement
# of kappa_z and of the corrected reduction over the one-step shifts and over the whole grid is computed here.
_ISP=os.environ.get("ALARM_INSAMPLE_LINK_ANALYSIS",f"{O}/LINK_ANALYSIS_insample.json")
if not os.path.exists(_ISP): raise SystemExit(f"{_ISP} is missing: the note of the policy tables states the measured sensitivity to selecting the cell map in sample, "
                                              "so the in-sample arm's LINK_ANALYSIS.json must be given (ALARM_INSAMPLE_LINK_ANALYSIS)")
_IS=json.load(open(_ISP))
if _IS[CH[0]]["transport_feature"].get("map_prespecified") is not False: raise SystemExit(f"{_ISP}: not the in-sample arm")
if D[CH[0]]["transport_feature"].get("map_prespecified") is not True: raise SystemExit(f"{O}/LINK_ANALYSIS.json: not the prespecified-map arm")
_mv=[]
for c in CH:
    for o in ("-1","1","2","3"):
        a=_IS[c]["transport_feature"]["by_offset"][o]; b=D[c]["transport_feature"]["by_offset"][o]
        _mv.append((D[c]["label"],o,abs(b["kappa_z"]-a["kappa_z"]),abs(b["corrected_reduction_z_pct"]-a["corrected_reduction_z_pct"])))
MAPSENS={"kz_plus1":max(m[2] for m in _mv if m[1]=="1"),"corr_plus1":max(m[3] for m in _mv if m[1]=="1"),
         "kz_any":max(m[2] for m in _mv),"corr_any":max(m[3] for m in _mv),
         "kz_any_channel":max(_mv,key=lambda m:m[2])[0],"corr_any_channel":max(_mv,key=lambda m:m[3])[0]}
_where=(f"both on {LONG[MAPSENS['kz_any_channel']]}" if MAPSENS["kz_any_channel"]==MAPSENS["corr_any_channel"]
        else f"on {LONG[MAPSENS['kz_any_channel']]} and {LONG[MAPSENS['corr_any_channel']]} respectively")
# One note, two documents. The %%TOKEN%% placeholders below are substituted with main-document
# labels for the main-text copy and with xr-prefixed ones for the supplement copy, so that neither
# copy can carry a reference the document it lands in cannot resolve.
MAIN_LABELS=(("%%THM%%","the identification result of the supplement"),("%%SEC%%","the calibration section"),
             ("%%LEVEL%%","the Supplementary Material"),("%%COR%%","the fixed-map inference"))
SUPP_LABELS=(("%%THM%%","the identification result"),
             ("%%SEC%%","the calibration section of the main text"),
             ("%%LEVEL%%",r"Section~\ref{s:level}"),("%%COR%%","the fixed-map inference"))
def relabel(s,pairs):
    for k,v in pairs: s=s.replace(k,v)
    return s
common_note=r"""Positive shifts loosen the limit (one step is """ + ", ".join(f"{UNIT[D[c]['label']]} on {TEX[D[c]['label']]}" for c in CH) + r"""); `replayed' is the
reduction $1-\hat B(u)/\hat B(u_0)$ in replayed excursion seconds that a crossing-based study would report, and each
`corrected' column divides the corresponding relative burden by $\hat\kappa$ (its interval follows from the interval of $\hat\kappa$). Duration transport carries the
annunciated-seconds function $\hat a_0(d)$ of %%THM%% to the shifted limit. Duration~$\times$~overshoot
transport carries the hybrid $\hat a^{\mathcal M}_0$ of the main text, with $z$ the excursion's maximal excess beyond the limit
in force (six bins): it equals $\hat a_0(d,z)$ on the retained joint cells $\mathcal M$ and the duration-marginal
$\hat{\bar a}_0(d)$ on cells holding fewer than $\textsc{nmin}=""" + f"{D[CH[0]]['transport_feature']['nmin']}" + r"""$ baseline excursions, so `coars.' is the share of the
estimated linked burden that falls on those coarsened cells, and the column lies closer to the duration route where that
share is large. The two routes differ only through the shift in the within-duration composition of
marginal excursions (%%SEC%%); neither is implied by the data at the limit in force.
The joint cell map is selected on the independent tuning subsample and applied unchanged, so the fixed-map hypothesis of
%%COR%% holds by construction; selecting it in sample instead --- the reported sensitivity --- moves
$\hat\kappa_z$ at the $+1$ step by at most $""" + f"{MAPSENS['kz_plus1']:.3f}" + r"""$ and the corrected reduction there by at most $""" + f"{MAPSENS['corr_plus1']:.2f}" + r"""$ percentage points,
the largest movement anywhere on this grid being $""" + f"{MAPSENS['kz_any']:.3f}" + r"""$ in $\hat\kappa_z$ and $""" + f"{MAPSENS['corr_any']:.1f}" + r"""$ percentage points in a corrected
reduction, """ + _where + r""".
`Tail' holds the linked seconds rather than the annunciation ratio fixed within the four tail bins ($\ge120$\,s).
The level route is the persistence-rule calibration of %%LEVEL%%; it has no solution on heart rate low.
Intervals are patient-cluster bootstraps ($B=400$) resampling every excursion, annunciation and counterfactual histogram
of a patient together. Four conditions are kept apart, each used in one sense only. \emph{Unsupported} is a failure of
identification condition~(S): the candidate charges a joint cell of zero baseline mass. \emph{Precision sparse} is a
reporting rule for cells below the twenty-excursion precision floor. \emph{Coarsening} is the $\textsc{nmin}$ rule above,
a declared convention and not evidence that the finer reward was identified there. \emph{Pooled-tail sensitive} marks a
shift placing a large share of its replayed seconds in the pooled bins at $\ge120$\,s.
$^{\dagger}$Pooled-tail sensitive, and reported for completeness rather than interpreted: $""" + f"{100*D['ecg-numLimit-heartRate-low']['support']['-1']['share_bin_ge600']:.0f}" + r"""\%$ of this shift's replayed
seconds fall in the $\ge600$\,s bin, which holds """ + f"{D['ecg-numLimit-heartRate-low']['support']['-1']['baseline_n_bin_ge600']}" + r""" excursions at the recorded limit, so the reward there rests on
extrapolation across a bin whose composition shifts. Its unsupported share is zero on the duration cells and $""" + f"{100*D['ecg-numLimit-heartRate-low']['support']['-1']['share_unsupported_joint']:.1f}" + r"""\%$ on
the joint cells."""
assert D['ecg-numLimit-heartRate-low']['support']['-1']['share_unsupported']==0.0, "the note says the unsupported share is zero on the duration cells"
# `outside_support` in analysis/transport_diagnostics.py is a tail-crowding flag
# (n_bin_ge600 > 3 x baseline and share_bin_ge600 > 0.2), not an identification-support flag; the
# reported unsupported share is zero at every shift. The dagger is named for what it measures.
main_note=relabel(common_note[:common_note.index("$^{\\dagger}$Pooled-tail sensitive")].rstrip(),MAIN_LABELS)
tab=r"""\begin{table}[htbp]\centering\scriptsize
\setlength{\tabcolsep}{2.8pt}
\caption{The one-step policy comparison corrected on the admissions of Table~\ref{tab:link}: the factor $\kappa$ by which
replay misstates the linked relative burden of a limit change, under the two declared transport scenarios of
Section~\ref{sec:ident}, with the tail and level-route sensitivities. The two scenario estimates are not bounds on the
truth (Section~\ref{sec:ident}); heart rate low is the least stable channel and its pooled-tail-sensitive tightening
contrast is reported in the Supplementary Material only.}
\label{tab:tworoutes}
""" + head + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\tightnote{""" + main_note + r"""}
\end{table}
"""
supp_note=relabel(common_note,SUPP_LABELS)
tab=r"""\begin{table}[htbp]\centering\scriptsize
\setlength{\tabcolsep}{2.4pt}
\caption{The policy comparison over the full grid of shifts ($-1$ to $+3$ steps), extending the one-step table of the main text
with the same conventions and intervals.}
\label{tab:tworoutesfull}
""" + head + "\n".join(rows_full) + r"""
\bottomrule
\end{tabular}
\tightnote{""" + supp_note + r""" Shifts beyond one step exceed most recorded within-admission
limit changes (Table~\ref{tab:limitchanges}) and are shown for the trend only.}
\end{table}
"""
emit("TABLE_TWO_ROUTES_FULL.tex",tab)
gaps=[]; agree=True
for c in CH:
    v=D[c]
    for o in ("1","2","3"):
        k=v["kappa"][o]
        if k["kappa_level"] is not None:
            gaps.append((v["label"],o,abs(k["corrected_level_pct"]-k["corrected_reduction_pct"])))
            if (k["kappa_level"]-1)*(k["kappa"]-1)<0: agree=False
summ["max_gap_pp_loosening"]=round(max(g[2] for g in gaps),2); summ["gaps"]=[(a,b,round(c,2)) for a,b,c in gaps]; summ["direction_agree_where_both"]=agree
# ---------------- Table: diagnostics (transport, support, reproducibility)
def suff_range(L):
    r=[DG["duration_sufficiency"][L][f"overshoot tertile|{i}"]["reward_ratio_duration_standardised"] for i in (0,1,2)]
    return r
rows=[]
for c in CH:
    v=D[c]; L=v["label"]; r=suff_range(L)
    tcm=DG["transport_comparison"][f"{L}|first change looser"]; lc=DG["limit_changes"][f"FIT p0|{L}"]; lc2=DG["limit_changes"][f"FIT p2|{L}"]
    rp=DG["reproducibility"][L]
    sp1=v["support"]["1"]; spm=v["support"]["-1"]
    def g(x,lo,hi): return f"{x:.2f} ({lo:.2f}--{hi:.2f})" if x is not None else "---"
    rows.append(f"{TEX[L]} & {r[0]:.2f} / {r[1]:.2f} / {r[2]:.2f} & {lc['admissions_with_change']+lc2['admissions_with_change']} & "
                f"{g(tcm['gamma_duration_only'],tcm['gamma_d_CI_lo'],tcm['gamma_d_CI_hi'])} & {g(tcm['gamma_feature'],tcm['gamma_z_CI_lo'],tcm['gamma_z_CI_hi'])} & "
                f"{100*sp1['share_cells_ge20_baseline']:.0f}\\% & {spm['n_bin_ge600']} / {spm['baseline_n_bin_ge600']} & "
                f"{rp['FIT p2']['kappa_+1_own']:.3f} / {rp['FIT p2']['kappa_z_+1_own']:.3f} & {rp['TUNE']['kappa_+1_on_FIT_durations']:.3f} \\\\".replace(",","\\,"))
tab=r"""\begin{table}[htbp]\centering\scriptsize
\setlength{\tabcolsep}{2.2pt}
\caption{Diagnostics for the transport, precision and reproducibility of the duration-transport correction.
\emph{Overshoot dependence}: the linked reward per excursion, standardized to the pooled duration distribution, in the
lowest, middle and highest tertile of maximal excess beyond the limit, relative to the pooled reward (1 = duration is
sufficient). \emph{Recorded changes}: admissions of the two fitting-split quarters with a recorded within-admission limit
change, and $\hat\gamma$ the observed after-change linked seconds over those predicted from the same admissions'
before-change reward under each transport convention (first change loosening, after-change excursions in duration cells
populated before the change; 1 = transport holds; Table~\ref{tab:limitchanges} gives the counts). \emph{Precision floor and pooled tail}: the share of replayed seconds under the
$+1$ shift falling in duration cells that hold at least 20 excursions at the recorded limit --- the twenty-excursion
precision floor, a reporting rule and not identification condition~(S) --- and the number of excursions
the $-1$ shift places in the $\ge600$\,s bin against the number there at the recorded limit, which is what makes that
shift pooled-tail sensitive. \emph{Reproducibility}:
$\hat\kappa(+1)$ under the two conventions on a disjoint fitting-split quarter, and the duration-transport
$\hat\kappa(+1)$ obtained by carrying the tuning-split reward function to the fitting-split counterfactual histograms.}
\label{tab:diag}
\begin{tabular}{@{}lcrllrrcc@{}}
\toprule
 & Overshoot & \multicolumn{3}{c}{Recorded limit changes} & \multicolumn{2}{c}{Precision / tail} & \multicolumn{2}{c}{Reproducibility, $\hat\kappa(+1)$} \\
\cmidrule(lr){2-2}\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Channel & T1 / T2 / T3 & adm. & $\hat\gamma_d$ (95\% CI) & $\hat\gamma_z$ (95\% CI) & $+1$ & $-1$: $n_{\ge600}$ & quarter B ($\hat\kappa$ / $\hat\kappa_z$) & tuning $\to$ fitting \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\tightnote{Intervals are patient-cluster bootstraps ($B=200$). Before/after comparisons within an admission are
confounded by whatever prompted the change, so $\hat\gamma$ tests the two conventions against each other more than
either against one; the respiration-rate row rests on """ + f"{DG['limit_changes']['FIT p0|RR low']['admissions_with_change']+DG['limit_changes']['FIT p2|RR low']['admissions_with_change']}" + r""" admissions.}
\end{table}
"""
emit("TABLE_DIAGNOSTICS.tex",tab)
summ["diag"]={D[c]["label"]:{"suff_overshoot":suff_range(D[c]["label"]),"gamma_d_looser":DG["transport_comparison"][f"{D[c]['label']}|first change looser"]["gamma_duration_only"],
                             "gamma_z_looser":DG["transport_comparison"][f"{D[c]['label']}|first change looser"]["gamma_feature"],
                             "kappa_p2":DG["reproducibility"][D[c]["label"]]["FIT p2"]["kappa_+1_own"],"kappa_z_p2":DG["reproducibility"][D[c]["label"]]["FIT p2"]["kappa_z_+1_own"],
                             "kappa_tune_on_fit":DG["reproducibility"][D[c]["label"]]["TUNE"]["kappa_+1_on_FIT_durations"]} for c in CH}
# ---------------- Table: level
rows=[]; isets={}
for c in CH:
    v=D[c]; L=v["label"]; lv=v["level"]; pf=v["persistence_form"]
    h0=np.array(v["hist_by_offset"]["0"]); h1=np.array(v["hist_by_offset"]["1"]); dm=np.array(v["psi_m_by_cell"]["d_mean_s"])
    NB=60; ts0=np.array(v["tail_secs_by_offset"]["0"]); ts1=np.array(v["tail_secs_by_offset"]["1"])
    dm0=dm.copy(); dm1=dm.copy()      # tail bins carry their own mean duration under each shift (exact replayed seconds)
    for b in range(4):
        if h0[NB+b]>0: dm0[NB+b]=ts0[b]/h0[NB+b]
        if h1[NB+b]>0: dm1[NB+b]=ts1[b]/h1[NB+b]
    def Delta(th): return float((h1@np.maximum(dm1-th,0))/(h0@np.maximum(dm0-th,0)))
    assert abs(100*(1-Delta(0.0))-v["kappa"]["1"]["replayed_reduction_pct"])<0.05, (L,100*(1-Delta(0.0)),v["kappa"]["1"]["replayed_reduction_pct"])
    ths=[pf["theta_rawmedian"],pf["theta_m_s"]]+([lv["theta_star_4s"]] if lv["theta_star_4s"]==lv["theta_star_4s"] else [])
    red=[100*(1-Delta(t)) for t in ths]; iset=(min(red),max(red)); isets[L]={"thetas":[round(t,1) for t in ths],"reduction_set_pct":[round(iset[0],1),round(iset[1],1)],
        "replayed_pct":v["kappa"]["1"]["replayed_reduction_pct"],"pair_pct":v["kappa"]["1"]["corrected_reduction_pct"]}
    ts=(f"{lv['theta_star_4s']:.1f}" if lv["theta_star_4s"]==lv["theta_star_4s"] else "none")
    rows.append(f"{TEX[L]} & {lv['rho4']:.2f} ({lv['CI']['rho4'][0]:.2f}--{lv['CI']['rho4'][1]:.2f}) & {lv['rho30']:.2f} ({lv['CI']['rho30'][0]:.2f}--{lv['CI']['rho30'][1]:.2f}) & "
                f"{lv['rho_measured_TEST_frozen']:.2f} ({TEST_CI[L][0]:.2f}--{TEST_CI[L][1]:.2f}) & {100*lv['unl4_share']:.0f}\\% & {lv['rho4_linked']:.2f} & "
                f"{pf['theta_rawmedian']} & {pf['theta_m_s']:.1f} & {ts} & {v['kappa']['1']['replayed_reduction_pct']:.1f}\\% & {iset[0]:.1f}--{iset[1]:.1f}\\% \\\\")
tab=r"""\begin{table}[htbp]\centering\scriptsize
\setlength{\tabcolsep}{2.6pt}
\caption{The level, on the admissions the pairs come from. $\hat\rho_4$ is replayed excursion seconds over
recorded annunciation seconds with both sides merged at 4\,s (the theory's convention), $\hat\rho_{30}$ the
same admissions under the 30-s convention, and `test' the held-out result, read once under the design fixed in advance, of
%%HOLDOUT%% (30-s convention, different admissions). `Unlinked' is the share of recorded
annunciation seconds attached to no excursion, and $\hat\rho_4^{\mathrm{linked}}$ the ratio with only
linked seconds in the denominator, which is $\mathbb E[D]/\mathbb E[\hat\psi_{\mathrm m}(D)]$ exactly.
Three fixed offsets summarize the device as a persistence rule: the median onset lag among linked
excursions (`raw', read under the 30-second convention of this column block; under the 4-second
convention it is %%RAW4%%\,s), the offset $\theta_{\mathrm m}$ that reproduces the linked annunciation seconds, and
the offset $\theta^\ast$ that reproduces $\hat\rho_4$ given $\hat q$. The last columns are the replayed
reduction at one loosening step and the range of corrected reductions over the three offsets: the
fixed-offset calibration range of %%LEVELSEC%% (the monotone offset map
evaluated at the offsets a persistence reading of the device would use; not an identified set).}
\label{tab:level}
\begin{tabular}{@{}lllllrrrrrr@{}}
\toprule
 & \multicolumn{3}{c}{Level $\rho$ (95\% CI)} & & & \multicolumn{3}{c}{Fixed offset (s)} & \multicolumn{2}{c}{Reduction, $+1$} \\
\cmidrule(lr){2-4}\cmidrule(lr){7-9}\cmidrule(lr){10-11}
Channel & $\hat\rho_4$ & $\hat\rho_{30}$ & test & Unl. & $\hat\rho_4^{\mathrm{lk}}$ & raw & $\theta_{\mathrm m}$ & $\theta^\ast$ & replayed & range \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\tightnote{Intervals are patient-cluster bootstraps ($B=400$). The three
offsets differ because the device behaves most nearly as a persistence device on respiration rate low,
where the annunciation ratio is near zero below $%%RRLAG%%$\,s and then rises sharply; on the other channels it
rises gradually and long excursions are annunciated only in part, which a single delay
constant fitted to the median lag cannot carry (%%LEVELSEC%%).}
\end{table}
"""
_raw4=[D[c]["lag_quantiles_linked"]["50"] for c in CH]          # the raw median under the 4-second convention, by channel
_RAW4=", ".join(f"${x:.0f}$" for x in _raw4[:-1])+f" and ${_raw4[-1]:.0f}$"
_RRLAG=f"{_raw4[0]:.0f}"
LEVEL_MAIN=(("%%HOLDOUT%%",r"Figure~\ref{fig:holdout}"),("%%LEVELSEC%%","the Supplementary Material"),("%%RAW4%%",_RAW4),("%%RRLAG%%",_RRLAG))
LEVEL_SUPP=(("%%HOLDOUT%%","the held-out figure of the main text"),
            ("%%LEVELSEC%%",r"Section~\ref{s:level}"),("%%RAW4%%",_RAW4),("%%RRLAG%%",_RRLAG))
emit("TABLE_LEVEL.tex",relabel(tab,LEVEL_MAIN))
emit("TABLE_LEVEL_SUPP.tex",relabel(tab,LEVEL_SUPP))
summ["calibration_ranges_plus1"]=isets
summ["kappa_pair_plus1"]={D[c]["label"]:D[c]["kappa"]["1"]["kappa"] for c in CH}
summ["kappa_z_plus1"]={D[c]["label"]:D[c]["transport_feature"]["by_offset"]["1"]["kappa_z"] for c in CH}
summ["kappa_z_minus1"]={D[c]["label"]:D[c]["transport_feature"]["by_offset"]["-1"]["kappa_z"] for c in CH}
summ["kappa_tail_secs_plus1"]={D[c]["label"]:D[c]["kappa"]["1"]["kappa_tail_seconds_fixed"] for c in CH}
summ["scenario_range_plus1"]={D[c]["label"]:sorted([D[c]["kappa"]["1"]["kappa"],D[c]["transport_feature"]["by_offset"]["1"]["kappa_z"]]) for c in CH}
summ["corrected_plus1_scenario_range_pct"]={D[c]["label"]:sorted([D[c]["kappa"]["1"]["corrected_reduction_pct"],D[c]["transport_feature"]["by_offset"]["1"]["corrected_reduction_z_pct"]]) for c in CH}
summ["replayed_plus1_pct"]={D[c]["label"]:D[c]["kappa"]["1"]["replayed_reduction_pct"] for c in CH}
summ["outside_support"]=[k for k,v in DG["support"].items() if v["outside_support"]]
summ["kappa_level_plus1"]={D[c]["label"]:D[c]["kappa"]["1"]["kappa_level"] for c in CH}
summ["kappa_pair_minus1"]={D[c]["label"]:D[c]["kappa"]["-1"]["kappa"] for c in CH}
summ["kappa_level_minus1"]={D[c]["label"]:D[c]["kappa"]["-1"]["kappa_level"] for c in CH}
summ["pair_CI_excludes_1_plus1"]={D[c]["label"]:bool(D[c]["kappa"]["1"]["CI"][0]>1) for c in CH}
summ["kappa_pair_range_loosening"]=[round(min(D[c]["kappa"][o]["kappa"] for c in CH for o in ("1","2","3")),3),round(max(D[c]["kappa"][o]["kappa"] for c in CH for o in ("1","2","3")),3)]
summ["max_pair_minus_replayed_pp_loosening"]=round(max(abs(D[c]["kappa"][o]["corrected_reduction_pct"]-D[c]["kappa"][o]["replayed_reduction_pct"]) for c in CH for o in ("1","2","3")),2)
json.dump(summ,open(f"{O}/LINK_SUMMARY.json","w"),indent=1); print(json.dumps(summ,indent=1))
# ---------------- Table: the main-text one-step comparison (TABLE_POLICY) and the implementation checks (TABLE_IMPL)
# Both are written from the same records as the tables above.
LONGNAME={"RR low":"Respiration rate, low","SpO2 low":r"\SpO{}, low","HR high":"Heart rate, high","HR low":"Heart rate, low"}
def r1h(x,places="0.1"):
    """Round half away from zero, as the hand-set tables did (40.25 -> 40.3, 0.985 -> 0.99)."""
    import decimal
    return str(decimal.Decimal(repr(x)).quantize(decimal.Decimal(places),rounding=decimal.ROUND_HALF_UP))
rows=[]
for c in CH:
    v=D[c]; L=v["label"]; k=v["kappa"]["1"]; z=v["transport_feature"]["by_offset"]["1"]; sp=v["support"]["1"]
    rng_=sorted([k["kappa"],z["kappa_z"]])
    rows.append(f"{LONGNAME[L]} & {k['replayed_reduction_pct']:.1f} & {r1h(k['corrected_reduction_pct'])} ({r1h(k['corrected_CI'][0])}--{r1h(k['corrected_CI'][1])}) & "
                f"{r1h(z['corrected_reduction_z_pct'])} ({r1h(z['corrected_z_CI'][0])}--{r1h(z['corrected_z_CI'][1])}) & {r1h(rng_[0],'0.01')}--{r1h(rng_[1],'0.01')} & "
                f"{v['level']['unl4_share']:.2f} & {sp['share_tail_ge120']:.2f} & {z['share_linked_burden_fallback']:.3f} \\\\")
rp=DG["reproducibility"]
_q=[rp[D[c]["label"]]["FIT p2"]["kappa_+1_own"] for c in CH]; _t=[rp[D[c]["label"]]["TUNE"]["kappa_+1_on_FIT_durations"] for c in CH]
tab=r"""\begin{table}[htbp]\centering\footnotesize
\setlength{\tabcolsep}{2.8pt}
\caption{The one-step loosening on the fitting-split subsample: replayed and calibrated reduction in
linked alarm burden, the span of the two transport scenarios, and what each estimate depends on.}
\label{tab:policy}
\begin{tabular}{@{}lrcccccc@{}}
\toprule
 & Replayed & \multicolumn{2}{c}{Calibrated reduction (95\% CI)} & Scenario & Unlinked & Pooled & Coars. \\
\cmidrule(lr){3-4}
Channel & (\%) & duration & dur.\ $\times$ overshoot & range $\kappa$ & share & tail & share \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\tightnote{Reduction is $100(1-B_{\mathrm m}(u_1)/B_{\mathrm m}(u_0))$ for the linked burden, under the
4-second convention, with 95\% patient-cluster bootstrap intervals ($B=400$). The scenario range is the
span of the two transport conventions fixed in advance, expressed as the factor $\kappa$ by which replay
misstates the relative burden; it is not a confidence interval and no result places the true value
inside it. The duration~$\times$~overshoot column estimates the hybrid reward $a^{\mathcal M}_0$ of
Section~\ref{sec:method}, which equals $a_0(d,z)$ on the retained joint cells and the duration-marginal
$\bar a_0(d)$ on cells holding fewer than $\textsc{nmin}=""" + f"{D[CH[0]]['transport_feature']['nmin']}" + r"""$ baseline excursions; `coars.\ share' is the
share of the estimated linked burden falling on those coarsened cells, and where it is large the column
lies closer to the duration route. The joint cell map is selected on the independent tuning subsample and
applied unchanged, so the fixed-map hypothesis of the inference holds by construction; selecting
it in sample instead --- the reported sensitivity --- moves $\hat\kappa_z$ at this shift by at most
$""" + f"{MAPSENS['kz_plus1']:.3f}" + r"""$ and the corrected reduction by at most $""" + f"{MAPSENS['corr_plus1']:.2f}" + r"""$ percentage points.
`Unlinked' is $q_{u_0}=U/(B_{\mathrm m}+U)$, the share of recorded annunciation seconds
attached to no excursion; `pooled tail' the share of replayed seconds above 120\,s, where the within-cell condition
is used. Re-estimating the duration-transport factor on a disjoint quarter of the fitting split
gives $""" + "$, $".join(f"{x:.2f}" for x in _q[:-1]) + "$ and $" + f"{_q[-1]:.2f}" + r"""$; with the tuning split's reward instead it is $""" + "$, $".join(f"{x:.2f}" for x in _t[:-1]) + "$\nand $" + f"{_t[-1]:.2f}" + r"""$. The four conditions named in Section~\ref{sec:limits} are used here in one sense each. Heart rate low
carries the widest intervals and the largest unlinked, pooled and coarsened shares, and its
scenario range straddles one; the one-step tightening on that channel is pooled-tail sensitive and is
reported in the Supplementary Material only.}
\end{table}
"""
assert D["ecg-numLimit-heartRate-low"]["kappa"]["1"]["kappa"]>1>D["ecg-numLimit-heartRate-low"]["transport_feature"]["by_offset"]["1"]["kappa_z"], "the note says heart rate low's scenario range straddles one"
emit("TABLE_POLICY.tex",tab)
# TABLE_IMPL: the in-sample arm's threshold sweep and empty-cell counts, the primary arm's counts in the caption
NBULK=D[CH[0]]["transport_feature"]["map_cells_total"]-4*D[CH[0]]["transport_feature"]["overshoot_bin_edges_in_steps"].__len__()
rows=[]
for c in CH:
    v=D[c]; w=_IS[c]; L=v["label"]; nm=w["transport_feature"]["kappa_z_by_nmin"]; e=w["transport_feature"]["empty_map_cell_events_in_bootstrap"]; lr=v["lr_order_diagnostic"]["1"]
    rows.append(f"{TEX[L]} & {nm['5']['1']:.3f} & {nm['10']['1']:.3f} & {nm['20']['1']:.3f} & {nm['50']['1']:.3f} & {w['transport_feature']['n_bulk_cells_fallback']} & "
                f"{e['replicates_with_any']} / {e['total']} & {lr['n_cells']} & {100*lr['share_burden_ratio_increasing']:.0f}\\% & {lr['spearman_ratio_vs_duration']:.2f} \\\\")
    assert abs(nm['5']['1']-w["transport_feature"]["by_offset"]["1"]["kappa_z"])<5e-4, (L,"the threshold-5 column is the in-sample estimate")
_pre_co=[D[c]["transport_feature"]["n_bulk_cells_fallback"] for c in CH]
_pre_em=[D[c]["transport_feature"]["empty_map_cell_events_in_bootstrap"] for c in CH]
_pre_sh=[100*e["total"]/(e["B"]*D[c]["transport_feature"]["map_cells_retained"]) for c,e in zip(CH,_pre_em)]
_B=_pre_em[0]["B"]
tab=r"""\begin{table}[htbp]\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{Two implementation checks for the one-step comparison. Left: the duration-by-overshoot scenario's
$\hat\kappa_z(+1)$ when the coarsening threshold --- the minimum number of excursions a duration-by-overshoot
cell must hold at the recorded limit before its own reward is used --- is set to 5 (the value fixed before any policy estimate was computed),
10, 20 or 50; the number of the """ + f"{NBULK}" + r""" bulk cells coarsened at the threshold of 5; and the number of the """ + f"{_B}" + r"""
bootstrap replicates in which at least one map cell was empty, with the total number of such cell events. This
block re-selects the map on the estimation sample at each threshold, so it is the in-sample sensitivity and not
the primary analysis: under the tuning-split map used throughout the paper, """ + ", ".join(str(x) for x in _pre_co[:-1]) + f" and {_pre_co[-1]}" + r""" bulk cells are
coarsened, and at least one map cell is empty in """ + ", ".join(f"{100*e['replicates_with_any']/e['B']:.0f}\\%" for e in _pre_em[:-1]) + f" and {100*_pre_em[-1]['replicates_with_any']/_pre_em[-1]['B']:.0f}\\%" + r""" of the replicates, those cells
taking the duration-marginal rate. Averaged over replicates, that map leaves """ + ", ".join(f"${x:.1f}\\%$" for x in _pre_sh[:-1]) + f" and ${_pre_sh[-1]:.1f}\\%$" + r"""
of its retained cells empty; a map selected off-sample need not have every cell occupied in every
resample, and the coarse rate carries those cells within the replicate. Right: the length-biased stochastic order of the replay-direction argument on
the replayed duration histograms at the recorded and the one-step-looser limit: the number of
2-second cells holding at least 20 excursions at both limits, the burden-weighted share of adjacent such
cells on which the histogram ratio $h_1(d)/h_0(d)$ increases, and the Spearman correlation of the ratio
with duration. The premise is not checked cell by cell; the stochastic order of the length-biased laws,
which the direction argument uses, is checked directly.}
\label{tab:impl}
\begin{tabular}{@{}lrrrrrcrrr@{}}
\toprule
 & \multicolumn{4}{c}{$\hat\kappa_z(+1)$ by coarsening threshold} & Cells & Empty-cell & \multicolumn{3}{c}{Likelihood-ratio diagnostic ($+1$)} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-6}\cmidrule(lr){7-7}\cmidrule(lr){8-10}
Channel & 5 & 10 & 20 & 50 & coarsened & replicates / events & cells & ratio rising & Spearman \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
emit("TABLE_IMPL.tex",tab)
