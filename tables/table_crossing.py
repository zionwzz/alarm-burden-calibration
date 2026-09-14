#!/usr/bin/env python3
"""Time from the numeric crossing to the recorded annunciation (stab:crossing), from the onset-lead
histograms comparison/c1_onset_hist_*.json written by scan/scan_onset_lead.py on the fitting set at
the naive setting. Every stripe is summed and the scan must be complete; quantiles are read off the
2-second histogram as the lower edge of the bin that reaches the quantile, the tails below and above
the histogram's range counted at their ends.

Writes TABLE_CROSSING.tex to the generated-tables directory and prints the numbers the text carries.
"""
from alarmreplay.paths import WORK as R, TABLES, ensure
ensure(TABLES)   # the generated tables' directory, created on demand
import collections, glob, json

CH = [("ecgResp-numLimit-respRate-low", "Respiration rate, low"), ("ecg-numLimit-heartRate-high", "Heart rate, high"),
      ("ecg-numLimit-heartRate-low", "Heart rate, low"), ("spO2-numLimit-satO2-low", r"\SpO{}, low")]
files = sorted(glob.glob(f"{R}/comparison/c1_onset_hist_*.json"))
if not files: raise SystemExit(f"{R}/comparison/c1_onset_hist_*.json is missing: run scan/scan_onset_lead.py")
H = {}; n = collections.Counter(); lo = collections.Counter(); hi = collections.Counter(); done = total = 0; meta = None
for fp in files:
    d = json.load(open(fp))
    if "hist" not in d: continue
    meta = d; done += d["admissions_done"]; total += d["admissions_total"]
    for ch, h in d["hist"].items():
        H[ch] = [a + b for a, b in zip(H[ch], h)] if ch in H else list(h)
        n[ch] += d["n_matched"][ch]; lo[ch] += d["lo_tail"][ch]; hi[ch] += d["hi_tail"][ch]
if meta is None or done < total: raise SystemExit(f"onset-lead scan incomplete: {done} of {total} admissions")
BASE, BW = meta["lo"], meta["bin_width_s"]


def q(ch, p):
    h = H[ch]; tot = sum(h) + lo[ch] + hi[ch]; acc = lo[ch]
    if acc >= p * tot: return BASE                        # inside the lower tail: report the histogram's floor
    for i, c in enumerate(h):
        acc += c
        if acc >= p * tot: return BASE + BW * i
    return BASE + BW * len(h)


def th(x): return f"{x:,}".replace(",", "\\,")


rows = []; nonneg = []
for ch, lab in CH:
    tot = sum(H[ch]) + lo[ch] + hi[ch]
    neg = lo[ch] + sum(c for i, c in enumerate(H[ch]) if BASE + BW * i < 0)
    nonneg.append(1 - neg / tot)
    rows.append(f"{lab:<22} & {th(n[ch]):>8} & {q(ch, .5):>2} & {q(ch, .25):>2}--{q(ch, .75):<2} & {q(ch, .1):>2} & {q(ch, .9):>2} \\\\")
tab = r"""\begin{table}[htbp]\centering\small
\caption{Time from the numeric crossing to the recorded annunciation.}
\label{stab:crossing}
\begin{tabular}{lrrrrr}
\toprule
Channel & Matched episodes & Median (s) & IQR (s) & 10th pct (s) & 90th pct (s) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\tightnote{Descriptive; computed on the whole fitting set (""" + th(done) + r""" admissions) at the naive setting so that
no additional read of the test set occurs. Positive values mean the monitor annunciated
after the numeric crossing. Between """ + f"{100*min(nonneg):.0f}" + r"""\% and """ + f"{100*max(nonneg):.0f}" + r"""\% of matched episodes had a
non-negative interval.}
\end{table}
"""
with open(f"{TABLES}/TABLE_CROSSING.tex", "w") as fh: fh.write(tab)
print("wrote TABLE_CROSSING.tex")
print(f"seeds table: whole fitting set, {th(done)} admissions, {th(sum(n.values()))} matched episodes; medians "
      + ", ".join(f"{lab} {q(ch, .5)}" for ch, lab in CH))
