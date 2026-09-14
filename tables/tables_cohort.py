#!/usr/bin/env python3
"""The descriptive tables of Part I of the Supplementary Material, from the epidemiology and
oncology-strata aggregates: cohort accounting (tab:cohort), recorded burden by alarm class
(tab:burden), burden by oncology tier and cancer group (stab:oncburden) and the channel-level
rates and silenced shares by stratum (stab:oncchannel).

The tables are written from the analysis outputs, and the prose numbers that accompany them are
printed so the text can be checked against the same source.

Inputs: alarm_epidemiology/A1_SUMMARY.json, sensitivity/ONCOLOGY_STRATA.json and
diagnostics/COHORT_COMPLETENESS.json.
"""
from alarmreplay.paths import WORK as R, TABLES, ensure
ensure(TABLES)   # the generated tables' directory, created on demand
import json, os

A = json.load(open(f"{R}/alarm_epidemiology/A1_SUMMARY.json"))
S = json.load(open(f"{R}/sensitivity/ONCOLOGY_STRATA.json"))["part1_burden_by_stratum"]
C = json.load(open(f"{R}/diagnostics/COHORT_COMPLETENESS.json"))


def th(n):
    """12345 -> 12\\,345"""
    s = f"{int(round(n)):,}"; return s.replace(",", "\\,")


def th1(x):
    """1234.5 -> 1\\,234.5"""
    s = f"{x:,.1f}"; return s.replace(",", "\\,")


def emit(name, text):
    with open(f"{TABLES}/{name}", "w") as fh: fh.write(text)
    print("wrote", name)


# ---------------------------------------------------------------- cohort accounting
n_cohort = C["cohort_admissions"]; n_scanned = A["admissions_scanned"]; n_nofile = A["no_alarm_file"]
n_norows = n_cohort - n_scanned - n_nofile
days = A["total_admission_days_spanned"]
splits = C["splits"]; pats = C["patients"]; n_pat = sum(pats.values())
tab = r"""\begin{table}[htbp]\centering\small
\caption{Cohort, recorded alarm data, and the patient-disjoint split.}
\label{tab:cohort}
\begin{tabular}{lrrr}
\toprule
 & Patients & Admissions & Admission-days \\
\midrule
Release cohort                                      & """ + th(n_pat) + " & " + th(n_cohort) + r""" & --- \\
\quad With recorded alarm rows               & ---    & """ + th(n_scanned) + " & " + th(days) + r""" \\
\quad Alarm file present, no alarm rows      & ---    & """ + th(n_norows) + r"""     & --- \\
\quad No alarm file                          & ---    & """ + str(n_nofile) + r"""       & --- \\
\addlinespace
Fitting set                                  & """ + th(pats["FIT"]) + " & " + th(splits["FIT"]) + r"""  & --- \\
Tuning set                                   & """ + th(pats["TUNE"]) + " & " + th(splits["TUNE"]) + r"""  & --- \\
Test set                                     & """ + th(pats["TEST"]) + " & " + th(splits["TEST"]) + r"""  & --- \\
\bottomrule
\end{tabular}
\tightnote{The split is at the patient level, allocated 50/25/25 by a
deterministic pseudorandom rule, and was fixed together with the candidate
parameter grid and every adequacy criterion before any model was fitted. Admission-days are spanned by recorded alarm rows (floor one
hour), on the calendar. Admissions with no alarm rows contribute neither burden nor denominator
and are reported, not imputed.}
\end{table}
"""
emit("TABLE_COHORT.tex", tab)

# ---------------------------------------------------------------- burden by class
CL = [("numLimit", "Numeric limit"), ("rhythm", "Arrhythmia"), ("tech", "Technical"), ("system", "System")]
runs_all = sum(A["by_class"][k]["annunciation_runs"] for k, _ in CL); secs_all = sum(A["by_class"][k]["alarm_seconds"] for k, _ in CL)
sil_all = sum(A["by_class"][k]["alarm_seconds"] * A["by_class"][k]["silenced_share_of_alarm_time"] for k, _ in CL) / secs_all
rows = []
for k, lab in CL:
    b = A["by_class"][k]
    rows.append(f"{lab:<13} & {th(b['annunciation_runs']):>12} & {b['runs_per_adm_day']:.1f}  & {b['alarm_min_per_adm_day']:.0f} & "
                f"{100*b['annunciation_runs']/runs_all:.0f}\\%  & {100*b['alarm_seconds']/secs_all:.0f}\\%  & {100*b['silenced_share_of_alarm_time']:.1f}\\% \\\\")
tab = r"""\begin{table}[htbp]\centering\small
\caption{Recorded alarm burden and silencing, by alarm class.}
\label{tab:burden}
\begin{tabular}{lrrrrrr}
\toprule
Alarm class & Annunciation & Runs per & Alarm-min per & Share of & Share of & Silenced share \\
            & runs         & adm-day  & adm-day       & runs     & alarm time & of alarm time \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
All classes   & """ + th(runs_all) + f" & {runs_all/days:.1f} & {secs_all/60/days:.0f} & 100\\% & 100\\% & {100*sil_all:.1f}\\% " + r"""\\
\bottomrule
\end{tabular}
\tightnote{""" + th(n_scanned) + " admissions, " + th(days) + r""" admission-days. An annunciation run is a
maximal sequence of alarm-active rows separated by no more than 30\,s. Silenced
share is silenced alarm-active seconds divided by total alarm-active seconds.}
\end{table}
"""
emit("TABLE_BURDEN.tex", tab)

# ---------------------------------------------------------------- burden by oncology tier and cancer group
TIERS = [("tier:strict_onc", "Strict oncology"), ("tier:onc_context_nonstrict", "Oncology context"), ("tier:non_onc", "Non-oncology")]
SITES = [("site:haematological", "Haematological"), ("site:thoracic", "Thoracic"), ("site:gastrointestinal", "Gastrointestinal"), ("site:other_solid", "Other solid")]
def brow(k, lab):
    v = S[k]; c = v["class_runs_per_adm_day"]
    return (f"{lab:<17} & {th(v['admissions']):>6} & {th1(v['admission_days']):>9} & {v['runs_per_adm_day']:.1f} & {c['numLimit']:>5.1f} & "
            f"{c['rhythm']:.1f} & {c['tech']:.1f} & {c['system']:.1f} & {100*v['silenced_share']:.1f}\\% \\\\")
tab = r"""\begin{table}[htbp]\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{Recorded alarm burden and silencing by oncology tier, and by cancer group within the strict-oncology
cohort (alarm-bearing admissions, full release).}
\label{stab:oncburden}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrrrrr}
\toprule
 & & & & \multicolumn{4}{c}{Runs per admission-day by class} & \\
\cmidrule(lr){5-8}
Stratum & Adm. & Adm.-days & Runs/day & Num.\ limit & Rhythm & Tech. & System & Silenced \\
\midrule
""" + "\n".join(brow(k, l) for k, l in TIERS) + r"""
\addlinespace
\multicolumn{9}{l}{\emph{Strict-oncology cohort by cancer group}} \\
""" + "\n".join(brow(k, l) for k, l in SITES) + r"""
\bottomrule
\end{tabular}
\tightnote{Tiers and cancer groups are admission-level classifications from the release's discharge diagnosis
fields (Section~\ref{s:cohort}); the release carries no facility or unit labels. Admission-days are the span of
the admission's alarm stream in the release (minimum one hour), so rates are per day of recorded alarm
stream; strict-oncology admissions have shorter recorded spans (""" + f"{S['tier:strict_onc']['admission_days']/S['tier:strict_onc']['admissions']:.2f}" + r""" days per admission against """ + f"{S['tier:non_onc']['admission_days']/S['tier:non_onc']['admissions']:.2f}" + r""" for
non-oncology admissions). Alarm-active hours per admission-day: """ + ", ".join(f"{S[k]['alarm_hours_per_adm_day']:.1f} ({w})" for (k, _), w in zip(TIERS, ("strict", "context", "non-oncology"))) + r""";
""" + " / ".join(f"{S[k]['alarm_hours_per_adm_day']:.1f}" for k, _ in SITES) + r""" for the four cancer groups in table order. Silenced share is the fraction of alarm-active time in a recorded silenced or
inactivated state, reported as device state without behavioural interpretation.}
\end{table}
"""
emit("TABLE_ONC_BURDEN.tex", tab)

# ---------------------------------------------------------------- channel-level rates and silenced shares by stratum
CHS = ["ecgResp-numLimit-respRate-low", "ecg-numLimit-heartRate-high", "ecg-numLimit-heartRate-low", "spO2-numLimit-satO2-low", "ecgResp-numLimit-respRate-high"]
def crow(k, lab):
    v = S[k]; r = v["channel_runs_per_adm_day"]; s = v["channel_silenced_share"]
    return f"{lab:<17} & " + " & ".join(f"{r[c]:>5.2f}" for c in CHS) + " & " + " & ".join(f"{s[c]:.3f}" for c in CHS) + r" \\"
tab = r"""\begin{table}[htbp]\centering\footnotesize
\setlength{\tabcolsep}{3.5pt}
\caption{Channel-level recorded burden (runs per admission-day) and silenced share of alarm-active time, by
oncology tier and by cancer group within the strict-oncology cohort.}
\label{stab:oncchannel}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lrrrrrrrrrr}
\toprule
 & \multicolumn{5}{c}{Runs per admission-day} & \multicolumn{5}{c}{Silenced share of alarm-active time} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-11}
Stratum & RR low & HR high & HR low & \SpO{} low & RR high & RR low & HR high & HR low & \SpO{} low & RR high \\
\midrule
""" + "\n".join(crow(k, l) for k, l in TIERS) + r"""
\addlinespace
""" + "\n".join(crow(k, l) for k, l in SITES) + r"""
\bottomrule
\end{tabular}
\tightnote{RR, respiration rate; HR, heart rate. Numeric-limit channels only; RR high is the channel excluded
from replay analyses. Denominators are as in Table~\ref{stab:oncburden}.}
\end{table}
"""
emit("TABLE_ONC_CHANNEL.tex", tab)

# ---------------------------------------------------------------- the numbers the Part I prose carries
top = A["top_alarms"]
print("\nPart I prose:")
print(f"  admissions with alarm rows {th(n_scanned)}, admission-days {th(days)}, no rows {n_norows}, no file {n_nofile}")
print(f"  runs per admission-day: all {runs_all/days:.0f}; " + ", ".join(f"{lab.lower()} {A['by_class'][k]['runs_per_adm_day']:.1f}" for k, lab in CL))
print(f"  respiration rate high {top['ecgResp-numLimit-respRate-high']['runs_per_adm_day']:.1f} runs per admission-day")
print(f"  silenced share overall {100*sil_all:.1f}%; by class " + ", ".join(f"{lab.lower()} {100*A['by_class'][k]['silenced_share_of_alarm_time']:.1f}%" for k, lab in CL)
      + "; tiers " + ", ".join(f"{100*A['by_tier'][t]['silenced_share']:.1f}%" for t in ("non_onc", "onc_context_nonstrict", "strict_onc")))
print(f"  SpO2 low silenced {100*top['spO2-numLimit-satO2-low']['silenced_share_of_alarm_time']:.1f}%, HR high {100*top['ecg-numLimit-heartRate-high']['silenced_share_of_alarm_time']:.1f}%")
print("  runs per admission-day by tier: " + ", ".join(f"{S[k]['runs_per_adm_day']:.0f}" for k, _ in TIERS)
      + f"; numeric-limit strict {S['tier:strict_onc']['class_runs_per_adm_day']['numLimit']:.1f} vs non-oncology {S['tier:non_onc']['class_runs_per_adm_day']['numLimit']:.1f}")
print("  SpO2 low runs/day by group: " + ", ".join(f"{l} {S[k]['channel_runs_per_adm_day']['spO2-numLimit-satO2-low']:.1f}" for k, l in SITES)
      + f"; haematological HR high {S['site:haematological']['channel_runs_per_adm_day']['ecg-numLimit-heartRate-high']:.1f}, silenced {100*S['site:haematological']['silenced_share']:.1f}%")
print("  silenced by channel, strict vs non-oncology: SpO2 low "
      f"{100*S['tier:strict_onc']['channel_silenced_share']['spO2-numLimit-satO2-low']:.1f}% vs {100*S['tier:non_onc']['channel_silenced_share']['spO2-numLimit-satO2-low']:.1f}%, HR high "
      f"{100*S['tier:strict_onc']['channel_silenced_share']['ecg-numLimit-heartRate-high']:.1f}% vs {100*S['tier:non_onc']['channel_silenced_share']['ecg-numLimit-heartRate-high']:.1f}%")
