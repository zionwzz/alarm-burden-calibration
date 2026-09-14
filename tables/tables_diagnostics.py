#!/usr/bin/env python3
"""Supplementary tables for the duration-sufficiency strata and for the recorded
within-admission limit changes with the two-scenario transport check.
"""
from alarmreplay.paths import WORK as R, TABLES, ensure
ensure(TABLES)   # the generated tables' directory, created on demand
import csv, json, os
D=f"{R}/diagnostics"; P=TABLES
TEX={"RR low":"RR low","SpO2 low":r"\SpO{} low","HR high":"HR high","HR low":"HR low"}
FEAT={"overshoot tertile":"maximal excess beyond the limit","entering-slope tertile":"entering slope (10-s change)","pre-excursion level tertile":"pre-excursion level (30-s mean)",
      "limit level tertile":"limit level","signal dropout (missing samples)":"dropped samples within the excursion"}
rows=[r for r in csv.DictReader(open(f"{D}/DURATION_SUFFICIENCY.csv"))]
out=[]
for ch in ["RR low","SpO2 low","HR high","HR low"]:
    first=True
    for f in FEAT:
        rr=[r for r in rows if r["channel"]==ch and r["feature"]==f]
        if not rr: continue
        cells=[]
        for lev in ("0","1","2"):
            x=[r for r in rr if r["stratum"]==lev]
            cells.append(f"{float(x[0]['reward_ratio_duration_standardised']):.2f} ({float(x[0]['CI_lo']):.2f}--{float(x[0]['CI_hi']):.2f})" if x else "---")
        n=sum(int(r["n_excursions"]) for r in rr)
        out.append(f"{TEX[ch] if first else ''} & {FEAT[f]} & {cells[0]} & {cells[1]} & {cells[2]} \\\\"); first=False
    out.append(r"\addlinespace")
out=out[:-1]
tab=r"""\begin{table}[htbp]\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{Is duration a sufficient statistic for the linked reward? For each feature fixed in advance the
excursions of the fitting-split phase-0 subsample are split into tertiles (or, for dropped samples, none
versus any), the reward-per-excursion function of each stratum is evaluated on the pooled duration weights,
and the ratio to the pooled reward is reported with a patient-cluster bootstrap interval ($B=200$); 1 means
the stratum's reward is fully explained by its durations. Limit level has two strata where the recorded
limits take few values.}
\label{tab:suff}
\begin{tabular}{@{}lllll@{}}
\toprule
Channel & Feature & Lowest tertile / none & Middle & Highest tertile / any \\
\midrule
""" + "\n".join(out) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(f"{P}/TABLE_SUFFICIENCY.tex","w").write(tab)
lc=[r for r in csv.DictReader(open(f"{D}/LIMIT_CHANGE_DISTRIBUTION.csv"))]
tc=[r for r in csv.DictReader(open(f"{D}/TRANSPORT_COMPARISON.csv"))]
out=[]
for ch in ["RR low","SpO2 low","HR high","HR low"]:
    a=[r for r in lc if r["channel"]==ch and r["set"]=="FIT p0"][0]; b=[r for r in lc if r["channel"]==ch and r["set"]=="FIT p2"][0]
    adm=int(a["admissions"])+int(b["admissions"]); adw=int(a["admissions_with_change"])+int(b["admissions_with_change"]); nch=int(a["n_changes"])+int(b["n_changes"])
    loos=(float(a["share_looser"])*int(a["n_changes"])+float(b["share_looser"])*int(b["n_changes"]))/max(nch,1)
    one=(float(a["share_one_step"])*int(a["n_changes"])+float(b["share_one_step"])*int(b["n_changes"]))/max(nch,1)
    two=(float(a["share_within_two_steps"])*int(a["n_changes"])+float(b["share_within_two_steps"])*int(b["n_changes"]))/max(nch,1)
    first=True
    for lab in ("all changes","first change looser","first change tighter"):
        t=[r for r in tc if r["channel"]==ch and r["comparison"]==lab][0]
        def g(x,lo,hi): return (f"{float(x):.2f} ({float(lo):.2f}--{float(hi):.2f})" if x not in ("",None) else "---")
        out.append((f"{TEX[ch]} & {adw} / {adm} ({100*adw/adm:.0f}\\%) & {nch} & {100*loos:.0f}\\% & {100*one:.0f}\\% / {100*two:.0f}\\%" if first else " & & & & ")+
                   f" & {lab} & {int(t['n_after_in_supported_cells']):,} & {g(t['gamma_duration_only'],t['gamma_d_CI_lo'],t['gamma_d_CI_hi'])} & {g(t['gamma_feature'],t['gamma_z_CI_lo'],t['gamma_z_CI_hi'])} & {g(t['predicted_ratio_feature_over_duration'],t['ratio_CI_lo'],t['ratio_CI_hi'])} \\\\".replace(",","\\,"))
        first=False
    out.append(r"\addlinespace")
out=out[:-1]
tab=r"""\begin{landscape}\begin{table}[htbp]\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{Recorded within-admission limit changes on the two fitting-split quarters, and the transport check
they allow. Left: admissions with at least one recorded change of the channel's limit over the admissions carrying the
channel, number of changes, share loosening, and the share of changes of exactly one step / at most two steps. Right: for
excursions after an admission's first change, the observed linked seconds over those predicted from the same
admissions' before-change reward under duration transport ($\hat\gamma_d$) and duration-by-overshoot transport
($\hat\gamma_z$), restricted to after-change excursions in duration cells populated before the change ($n$),
and the ratio of the two predictions; patient-cluster bootstrap intervals ($B=200$).}
\label{tab:limitchanges}
\begin{tabular}{@{}lrrrrlrlll@{}}
\toprule
Channel & With change / adm. & Changes & Looser & 1 step / $\le2$ & Comparison & $n$ & $\hat\gamma_d$ (95\% CI) & $\hat\gamma_z$ (95\% CI) & $\hat P_z/\hat P_d$ (95\% CI) \\
\midrule
""" + "\n".join(out) + r"""
\bottomrule
\end{tabular}
\tightnote{Off/on events (a limit recorded as switched off or back on) are excluded from the change counts.
Before/after comparisons within an admission are confounded by whatever prompted the change; the last three
columns test the two conventions against each other more than either against one.}
\end{table}\end{landscape}
"""
open(f"{P}/TABLE_LIMITCHANGES.tex","w").write(tab); print("ok")
