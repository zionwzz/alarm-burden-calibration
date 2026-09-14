#!/usr/bin/env python3
"""Size-matched random cohorts, as the control for a restricted arm.

A restricted arm differs from the primary analysis in two ways at once: the patients are
different, and there are far fewer of them. This draws random admission cohorts of the same size
as the restriction, runs the same estimator on each, and records the two quantities that decide
whether an arm is interpretable -- the share of candidate replayed time in duration-by-overshoot
cells with no baseline excursion, and the width of the correction-factor interval. Against that
reference, a restricted arm can be said to lose overlap because of who is in it rather than how
many.

Writes one row per draw and channel to <work>/diagnostics/MATCHED_SIZE_CONTROL.csv.
Nothing patient-level is written; the drawn cohorts are not kept.
"""
import argparse, csv, json, os, random, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from alarmreplay.paths import WORK as R, COHORT as SAMP  # noqa: E402

CH = ["ecgResp-numLimit-respRate-low", "spO2-numLimit-satO2-low",
      "ecg-numLimit-heartRate-high", "ecg-numLimit-heartRate-low"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True, help="admissions per draw")
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--shift", default="1")
    ap.add_argument("--out", default="MATCHED_SIZE_CONTROL.csv", help="file name under <work>/diagnostics")
    a = ap.parse_args()

    allad = sorted({r["aCSN"] for r in csv.DictReader(open(SAMP))}, key=int)
    if a.size > len(allad):
        raise SystemExit(f"asked for {a.size} of {len(allad)} admissions")
    rng = random.Random(a.seed)
    out = os.path.join(R, "diagnostics", a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["draw", "channel", "n_admissions", "n_excursions",
                    "share_unsupported_joint_pct", "kappa_z", "kappa_z_ci_width"])
        for d in range(a.draws):
            keep = rng.sample(allad, a.size)
            with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as tf:
                ww = csv.writer(tf); ww.writerow(["aCSN"])
                for c in keep: ww.writerow([c])
                path = tf.name
            env = dict(os.environ, ALARM_COHORT_SUBSET=path,
                       ALARM_COHORT_LABEL=f"size-matched random draw {d}",
                       ALARM_LINK_TAG="_matched")
            p = subprocess.run([sys.executable, "analysis/link_analysis.py"], cwd=HERE, env=env,
                               capture_output=True, text=True)
            os.unlink(path)
            if p.returncode:
                w.writerow([d, "ALL", "", "", "run failed", p.stderr.strip().splitlines()[-1][:120], ""])
                f.flush(); print(f"draw {d}: failed", flush=True); continue
            P = json.load(open(os.path.join(R, "policy", "LINK_ANALYSIS_matched.json")))
            for ch in CH:
                z = P[ch]["transport_feature"]["by_offset"][a.shift]
                w.writerow([d, P[ch]["label"], P[ch]["n_admissions"], P[ch]["n_excursions"],
                            round(100 * P[ch]["support"][a.shift]["share_unsupported_joint"], 3),
                            round(z["kappa_z"], 4), round(z["CI"][1] - z["CI"][0], 4)])
            f.flush(); print(f"draw {d}: done", flush=True)
    print("wrote", os.path.basename(out))


if __name__ == "__main__":
    main()
