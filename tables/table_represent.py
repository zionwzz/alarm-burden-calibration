#!/usr/bin/env python3
"""Representativeness of the two every-fourth-admission subsamples against the full fitting split
(tab:represent), from diagnostics/SUBSAMPLE_REPRESENTATIVENESS.csv written by
analysis/transport_diagnostics.py.
"""
from alarmreplay.paths import WORK as R, TABLES, ensure
ensure(TABLES)   # the generated tables' directory, created on demand
import csv

SETS = [("full FIT", "Full fitting split"), ("every-4th subsample (phase 0)", "Quarter A (analysis subsample)"),
        ("every-4th subsample (phase 2)", "Quarter B (reproducibility)")]
CHN = {"full FIT": "full FIT", "every-4th subsample (phase 0)": "subsample p0", "every-4th subsample (phase 2)": "subsample p2"}
CH = ["RR low", "SpO2 low", "HR high", "HR low"]
head = {}; per = {}
with open(f"{R}/diagnostics/SUBSAMPLE_REPRESENTATIVENESS.csv") as fh:
    for row in csv.reader(fh):
        if not row: continue
        if row[0] in dict(SETS) and row[1] == "admissions":
            tiers = dict(x.split(":") for x in row[5:])
            head[row[0]] = (int(row[2]), int(row[4]), tiers)
        elif row[0] in CH and row[1] in CHN.values():
            kv = dict(x.split("=") for x in row[2:])
            per[(row[0], row[1])] = kv


def th(n): return f"{n:,}".replace(",", "\\,")


rows = []
for key, lab in SETS:
    adm, pat, tiers = head[key]
    cells = []
    for ch in CH:
        kv = per[(ch, CHN[key])]
        cells.append(f"{float(kv['mean_D']):.1f} / {100*float(kv['share_ge60s']):.1f} / {float(kv['secs_per_adm'])/60:.0f}")
    rows.append(f"{lab} & {th(adm)} & {th(pat)} & {100*float(tiers['non_onc']):.0f} / {100*float(tiers['onc_context_nonstrict']):.0f} / {100*float(tiers['strict_onc']):.0f} & " + " & ".join(cells) + r" \\")
tab = r"""\begin{table}[htbp]\centering\scriptsize
\setlength{\tabcolsep}{2.2pt}
\caption{Representativeness of the every-fourth-admission subsamples against the full fitting split: admissions,
patients, diagnosis-derived oncology tier shares (non-oncology / oncology context / strict oncology, \%), and per
channel the mean excursion duration (s) / share of excursions of at least a minute (\%) / replayed excursion
minutes per admission at the recorded limit (4-s runs; full-split marginal replay at the recorded limit).}
\label{tab:represent}
\begin{tabular}{@{}lrrcllll@{}}
\toprule
 & & & Tier shares & \multicolumn{4}{c}{Mean $D$ / $\ge60$\,s / min per adm.} \\
\cmidrule(lr){5-8}
Set & Adm. & Patients & (\%) & RR low & \SpO{} low & HR high & HR low \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
with open(f"{TABLES}/TABLE_REPRESENT.tex", "w") as fh: fh.write(tab)
print("wrote TABLE_REPRESENT.tex")
