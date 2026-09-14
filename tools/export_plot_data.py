#!/usr/bin/env python3
"""Export the aggregates the figures need from the working tree of a credentialed run.

The data figures are rendered from these files alone. The exported files are cell- and channel-level
aggregates: no patient or admission identifier is written, and no row is small enough to identify an
admission; the smallest unit exported is a duration cell pooled over patients, with its excursion
count. They are products of the credentialed analysis and are not distributed with the code.

Run from the repository root with the working tree present:

    ALARM_WORK=/path/to/work python3 tools/export_plot_data.py

Files written to ALARM_PLOTDATA (default generated/plot_data):

    reward_by_cell.csv        duration cell, mean duration, excursions, linked seconds per
                              excursion, annunciation ratio and its bootstrap band,
                              annunciation probability                              (Figure 4)
    reward_by_overshoot.csv   annunciation ratio by duration group and overshoot class,
                              with the excursions behind each                       (Figure 4)
    lag_curves.csv            observed lag ECDF by channel
    policy_change.csv         replayed and corrected percentage change in linked burden by
                              channel and shift, with intervals and support         (Figure 5)
    holdout_validation.csv    held-out burden ratios and validation criteria        (Figure 3)
    design.json               cohort and split counts, and the analysis subsample   (Figure S1)

`holdout_validation.csv` comes from the held-out adjudication of the delay model, a single read of
the held-out split: analysis/adjudicate_test.py writes delay_model/A2_TEST_RESULTS.json and
comparison/A3_NAIVE_VS_DELAY_AWARE.csv, and the exporter builds the rows from those two files. It
fails if they are absent; it never keeps an earlier copy.
"""
from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHANNELS = [
    ("ecgResp-numLimit-respRate-low", "RR low", "Respiration rate, low"),
    ("spO2-numLimit-satO2-low", "SpO2 low", "SpO2, low"),
    ("ecg-numLimit-heartRate-high", "HR high", "Heart rate, high"),
    ("ecg-numLimit-heartRate-low", "HR low", "Heart rate, low"),
]
OVERSHOOT_CLASS = {"0": "at the limit", "1": "one step beyond", "2+": "two or more steps"}


def _load(work, name, subdir="policy"):
    path = os.path.join(work, subdir, name)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {os.path.relpath(path)}  ({len(rows)} rows)")


def export(work, out):
    os.makedirs(out, exist_ok=True)
    LA = _load(work, "LINK_ANALYSIS.json")
    LS = _load(work, "LINK_SUMMARY.json")
    OC = _load(work, "OVERSHOOT_CURVES.json")
    if LA is None:
        sys.exit(f"LINK_ANALYSIS.json not found under {work}/policy; run analysis/link_analysis.py first")

    # ---- Figure 3: the reward by duration cell -------------------------------------------------
    rows = []
    for key, short, _ in CHANNELS:
        v = LA[key]["psi_m_by_cell"]
        n_exact = len(v["d_mean_s"]) - len(v["tail_bins_s"])
        for c, (d, psi, n, lo, hi, p) in enumerate(zip(v["d_mean_s"], v["psi_m_s"], v["n"],
                                                       v["h_CI_lo"], v["h_CI_hi"], v["ann_prob"])):
            if not n:
                continue
            rows.append([short, c, "exact" if c < n_exact else "pooled", f"{d:.2f}", n,
                         f"{psi:.4f}", "" if d <= 0 else f"{psi / d:.5f}",
                         f"{lo:.5f}", f"{hi:.5f}", "" if p is None else f"{p:.4f}"])
    write_csv(os.path.join(out, "reward_by_cell.csv"),
              ["channel", "cell", "cell_kind", "mean_duration_s", "excursions",
               "linked_seconds_per_excursion", "annunciation_ratio", "ratio_ci_lo", "ratio_ci_hi",
               "annunciation_probability"], rows)

    # ---- Figure 3: the reward by overshoot class -----------------------------------------------
    rows = []
    if OC is not None:
        for key, short, _ in CHANNELS:
            for cls, series in OC[key].items():
                for d, h, n in zip(series.get("d_s", []), series.get("h", []), series.get("n", [])):
                    if n:
                        rows.append([short, OVERSHOOT_CLASS.get(cls, cls), f"{d:.1f}", f"{h:.4f}", n])
    if not rows:
        for key, short, _ in CHANNELS:
            b = LA[key]["transport_feature"].get("baseline_reward_by_overshoot_bin")
            if not b:
                continue
            for gi, group in enumerate(b["cells"]):
                for bi, (h, n) in enumerate(zip(b["h"][gi], b["n"][gi])):
                    if h is not None:
                        rows.append([short, group, bi, f"{h:.4f}", n])
        write_csv(os.path.join(out, "reward_by_overshoot.csv"),
                  ["channel", "duration_group", "overshoot_bin", "annunciation_ratio", "excursions"],
                  rows)
    else:
        write_csv(os.path.join(out, "reward_by_overshoot.csv"),
                  ["channel", "overshoot_class", "duration_s", "annunciation_ratio", "excursions"],
                  rows)

    # ---- Figure 1: the lag law ----------------------------------------------------------------
    rows = []
    for key, short, _ in CHANNELS:
        lc = LA[key]["lag_curves"]
        for x, e in zip(lc["x_s"], lc["observed_ecdf"]):
            rows.append([short, f"{x:.0f}", f"{e:.4f}"])
    write_csv(os.path.join(out, "lag_curves.csv"),
              ["channel", "lag_s", "observed_ecdf"], rows)

    # ---- Figure 4: the policy consequence ------------------------------------------------------
    rows = []
    for key, short, _ in CHANNELS:
        v = LA[key]
        unsupported = set()
        if LS and "outside_support" in LS:
            unsupported = {s for s in LS["outside_support"]}
        for off, k in v["kappa"].items():
            if off == "0":
                continue
            z = v["transport_feature"]["by_offset"].get(off, {})
            rows.append([
                short, off,
                f"{k['replayed_reduction_pct']:.2f}",
                f"{k['corrected_reduction_pct']:.2f}",
                f"{k['corrected_CI'][0]:.2f}", f"{k['corrected_CI'][1]:.2f}",
                "" if "corrected_reduction_z_pct" not in z else f"{z['corrected_reduction_z_pct']:.2f}",
                "" if "corrected_z_CI" not in z else f"{z['corrected_z_CI'][0]:.2f}",
                "" if "corrected_z_CI" not in z else f"{z['corrected_z_CI'][1]:.2f}",
                "" if k.get("corrected_level_pct") is None else f"{k['corrected_level_pct']:.2f}",
                f"{v['support'][off]['share_unsupported']:.4f}",
                f"{v['support'][off]['share_tail_ge120']:.4f}",
                "yes" if f"{short}|{off}" in unsupported else "no",
            ])
    write_csv(os.path.join(out, "policy_change.csv"),
              ["channel", "shift_steps", "replayed_change_pct", "corrected_duration_pct",
               "corrected_duration_ci_lo", "corrected_duration_ci_hi", "corrected_overshoot_pct",
               "corrected_overshoot_ci_lo", "corrected_overshoot_ci_hi", "corrected_level_pct",
               "share_unsupported", "share_pooled_tail", "outside_support"], rows)

    # ---- the design ----------------------------------------------------------------------------
    design = {
        "channels": {short: {"excursions": LA[key]["n_excursions"],
                             "linked": LA[key]["n_linked"],
                             "patients": LA[key]["n_patients"],
                             "admissions": LA[key]["n_admissions"],
                             "link_share": round(LA[key]["link_share"], 4),
                             "plateau_q": round(LA[key]["level"]["q_plateau"], 4),
                             "unlinked_share": round(LA[key]["level"]["unl4_share"], 4)}
                     for key, short, _ in CHANNELS},
        "bootstrap_replicates": LA[CHANNELS[0][0]]["n_bootstrap"],
    }
    # cohort accounting, for the study-flow figure and the burden figure's header: the epidemiology
    # summary and the completeness report of the scan tree, both aggregate
    a1p = os.path.join(work, "alarm_epidemiology", "A1_SUMMARY.json"); ccp = os.path.join(work, "diagnostics", "COHORT_COMPLETENESS.json")
    if os.path.exists(a1p) and os.path.exists(ccp):
        A1 = json.load(open(a1p)); CC = json.load(open(ccp))
        runs = sum(v["annunciation_runs"] for v in A1["by_class"].values())
        design["cohort"] = {"admissions": CC["cohort_admissions"], "patients": sum(CC["patients"].values()),
                            "admissions_with_alarm_rows": A1["admissions_scanned"], "no_alarm_file": A1["no_alarm_file"],
                            "alarm_file_without_rows": CC["cohort_admissions"] - A1["admissions_scanned"] - A1["no_alarm_file"],
                            "admission_days": A1["total_admission_days_spanned"], "annunciation_runs": runs,
                            "runs_per_admission_day": round(runs / A1["total_admission_days_spanned"], 1),
                            "splits": {s: {"patients": CC["patients"][s], "admissions": CC["splits"][s]} for s in ("FIT", "TUNE", "TEST")}}
    else:
        raise SystemExit("alarm_epidemiology/A1_SUMMARY.json and diagnostics/COHORT_COMPLETENESS.json are needed for the cohort block of design.json")
    with open(os.path.join(out, "design.json"), "w") as f:
        json.dump(design, f, indent=1)
    print(f"  wrote {os.path.relpath(os.path.join(out, 'design.json'))}")

    # ---- Figure 2: the held-out adjudication ---------------------------------------------------
    # analysis/adjudicate_test.py writes delay_model/A2_TEST_RESULTS.json (the selected model's
    # held-out recall, burden ratio, onset agreement, intervals and criteria) and
    # comparison/A3_NAIVE_VS_DELAY_AWARE.csv (the naive comparator with its interval). The rows
    # are built from those two files; if either is absent the export fails rather than keeping
    # an earlier copy of the CSV.
    res_path = os.path.join(work, "delay_model", "A2_TEST_RESULTS.json")
    a3_path = os.path.join(work, "comparison", "A3_NAIVE_VS_DELAY_AWARE.csv")
    if not (os.path.isfile(res_path) and os.path.isfile(a3_path)):
        raise SystemExit(f"held-out adjudication outputs missing ({res_path}, {a3_path}): run analysis/adjudicate_test.py")
    with open(res_path) as f:
        res = json.load(f)
    a3 = {}
    with open(a3_path, newline="") as f:
        for r in csv.DictReader(f):
            lo, hi = (float(x) for x in r["naive_CI"].strip("[]").split(","))
            a3[r["channel"]] = (float(r["naive_burden_ratio"]), lo, hi)
    rows = []
    for key, short, _ in CHANNELS:
        c = res["channels"][key]; nv = a3[key]
        rows.append([short, f"{nv[0]:.2f}", f"{nv[1]:.2f}", f"{nv[2]:.2f}", c["tau"], c["g"],
                     f"{c['TEST']['recall']:.3f}", f"{c['recall_CI'][0]:.3f}", f"{c['recall_CI'][1]:.3f}",
                     f"{c['TEST']['burden_ratio']:.2f}", f"{c['burden_CI'][0]:.2f}", f"{c['burden_CI'][1]:.2f}",
                     f"{c['TEST']['onset_within10']:.3f}", sum(1 for v in c["criteria"].values() if v)])
    write_csv(os.path.join(out, "holdout_validation.csv"), ["channel", "naive_ratio", "naive_ci_lo", "naive_ci_hi", "tau_s", "gap_s",
                     "episode_recall", "recall_ci_lo", "recall_ci_hi", "fitted_ratio",
                     "fitted_ci_lo", "fitted_ci_hi", "onset_within_10s", "criteria_met"], rows)

if __name__ == "__main__":
    from alarmreplay.paths import WORK, PLOTDATA
    work = os.environ.get("ALARM_WORK", WORK)
    out = PLOTDATA
    print(f"exporting plot data from {work} to {out}")
    export(work, out)
    print("done. These files contain no patient or admission identifiers.")
