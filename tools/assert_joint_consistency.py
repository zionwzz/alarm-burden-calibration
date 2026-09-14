#!/usr/bin/env python3
"""The duration-by-overshoot cell identity between the two scanners that see the same excursions.

scan_linkage_features.py writes one row per excursion at the recorded limit, with its duration
and its overshoot; scan_counterfactual_joint.py writes, per admission and channel, the joint
duration-by-overshoot histogram of the same excursions at offset zero. Under one convention the
two must agree exactly: for every patient, admission, channel, duration cell and overshoot bin,
the same number of excursions and the same number of seconds. This script asserts that, and
refuses to pass on anything that could make the agreement vacuous:

  * empty inputs on either side;
  * an admission that appears in more than one stripe file, or a replay row repeated;
  * a stripe whose state file does not show every admission of its intended list processed;
  * a stripe file missing from the contiguous set 0..N-1;
  * a feature scan without an overshoot column, or a replay scan without its bin edges;
  * a replay row whose own totals disagree with its cells.

Only aggregate counts are printed or written to the summary; the patient-level list of
mismatching cells goes to a log inside the working tree.

usage: assert_joint_consistency.py --baseline-tag FIT4p0
       assert_joint_consistency.py --baseline-tag FIT4p2 --replay-pattern 'pd3_rows_p2_[0-9]*.csv'
       assert_joint_consistency.py --baseline-tag TUNE2p0 --replay-pattern 'pd3_rows_TUNE2p0_*.csv'
"""
import argparse, bisect, collections, csv, glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NB = 60; TAILB = [120, 180, 300, 600]
OBE = [0, 1, 2, 4, 8, 16]
STEP = {"ecg-numLimit-heartRate-high": 5.0, "ecg-numLimit-heartRate-low": 5.0,
        "spO2-numLimit-satO2-low": 1.0, "ecgResp-numLimit-respRate-low": 1.0}


def parse_tag(tag):
    m = re.fullmatch(r"([A-Z]+)(\d+)p(\d+)", tag)
    if not m:
        raise SystemExit(f"cannot read split/subsample/phase from tag {tag!r}")
    return m.group(1), int(m.group(2)), int(m.group(3))


def intended(cohort, split_csv, split, sub, phase, nstr):
    """The admission list of each stripe, exactly as the scanners build it."""
    splitmap = {r["aMRN"]: r["split"] for r in csv.DictReader(open(split_csv))}
    adms = [(r["aMRN"], r["aCSN"]) for r in csv.DictReader(open(cohort)) if splitmap.get(r["aMRN"]) == split]
    adms = [a for i, a in enumerate(adms) if i % sub == phase]
    return [[a for i, a in enumerate(adms) if i % nstr == k] for k in range(nstr)]


def stripes_of(files, prefix):
    ks = []
    for fp in files:
        m = re.search(prefix + r"(\d+)\.csv$", os.path.basename(fp))
        if m: ks.append(int(m.group(1)))
    return sorted(ks)


def cell_of(d):
    return d // 2 if d < 120 else NB + bisect.bisect_right(TAILB, d) - 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=os.environ.get("ALARM_WORK"), help="working tree (default $ALARM_WORK)")
    ap.add_argument("--cohort", default=os.environ.get("ALARM_COHORT"), help="cohort table (default $ALARM_COHORT)")
    ap.add_argument("--baseline-tag", required=True, help="tag of the feature scan, e.g. FIT4p0")
    ap.add_argument("--replay-pattern", default=None, help="glob of the joint replay rows under policy/ (default from the tag)")
    ap.add_argument("--stripes", type=int, default=None, help="expected number of stripes of the feature scan (default: as found)")
    ap.add_argument("--replay-stripes", type=int, default=None, help="expected number of replay stripes (default --stripes)")
    ap.add_argument("--out", default=None, help="JSON summary (default diagnostics/JOINT_IDENTITY_<tag>.json)")
    ap.add_argument("--mismatch-log", default=None, help="patient-level mismatch log (default diagnostics/JOINT_IDENTITY_<tag>_mismatches.csv)")
    a = ap.parse_args()
    if not a.work or not a.cohort:
        raise SystemExit("--work and --cohort (or ALARM_WORK and ALARM_COHORT) are required")
    W = a.work; tag = a.baseline_tag; split, sub, phase = parse_tag(tag)
    split_csv = os.path.join(W, "design", "SPLIT_ASSIGNMENT.csv")
    if a.replay_pattern is None:
        a.replay_pattern = "pd3_rows_[0-9]*.csv" if tag == "FIT4p0" else ("pd3_rows_p2_[0-9]*.csv" if tag == "FIT4p2" else f"pd3_rows_{tag}_*.csv")
    # the replay pattern must name the same sample as the tag
    ptag = "" if tag == "FIT4p0" else ("p2_" if tag == "FIT4p2" else f"{tag}_")
    if not a.replay_pattern.startswith(f"pd3_rows_{ptag}"):
        raise SystemExit(f"replay pattern {a.replay_pattern!r} does not name the sample of tag {tag}")
    fails = []; notes = []

    # ---- files and completeness
    lk_files = sorted(glob.glob(os.path.join(W, "policy", f"lk2_rows_{tag}_*.csv")))
    rp_files = sorted(glob.glob(os.path.join(W, "policy", a.replay_pattern)))
    rp_files = [f for f in rp_files if re.fullmatch(rf"pd3_rows_{re.escape(ptag)}\d+\.csv", os.path.basename(f))]
    lk_k = stripes_of(lk_files, rf"lk2_rows_{re.escape(tag)}_"); rp_k = stripes_of(rp_files, rf"pd3_rows_{re.escape(ptag)}")
    n_lk = a.stripes if a.stripes is not None else len(lk_k)
    n_rp = a.replay_stripes if a.replay_stripes is not None else (a.stripes if a.stripes is not None else len(rp_k))
    if not lk_files: fails.append("no feature-scan files: empty input")
    if not rp_files: fails.append("no replay files: empty input")
    exp_lk = intended(a.cohort, split_csv, split, sub, phase, n_lk); exp_rp = intended(a.cohort, split_csv, split, sub, phase, n_rp)
    # a stripe with no intended admission writes no file; every other stripe must be present
    need_lk = [k for k in range(n_lk) if exp_lk[k]]; need_rp = [k for k in range(n_rp) if exp_rp[k]]
    if [k for k in lk_k if k < n_lk] != need_lk or any(k >= n_lk for k in lk_k): fails.append(f"feature-scan stripes present {lk_k}, expected {need_lk}")
    if [k for k in rp_k if k < n_rp] != need_rp or any(k >= n_rp for k in rp_k): fails.append(f"replay stripes present {rp_k}, expected {need_rp}")
    if fails:
        return finish(a, tag, fails, notes, None)
    for k in need_lk:
        st = os.path.join(W, "cache", f"lk2_state_{tag}_{k}.json")
        i = json.load(open(st))["i"] if os.path.exists(st) else -1
        if i < len(exp_lk[k]): fails.append(f"feature-scan stripe {k} incomplete: state {i} of {len(exp_lk[k])} admissions")
    for k in need_rp:
        st = os.path.join(W, "cache", f"pd3_state_{ptag}{k}.json")
        i = json.load(open(st))["i"] if os.path.exists(st) else -1
        if i < len(exp_rp[k]): fails.append(f"replay stripe {k} incomplete: state {i} of {len(exp_rp[k])} admissions")
    meta = [f for f in glob.glob(os.path.join(W, "policy", f"pd3_rows_{ptag}*_meta.json"))]
    if not meta or any("overshoot_bin_edges_in_steps" not in json.load(open(m)) for m in meta):
        fails.append("replay scan carries no overshoot bin edges")
    else:
        edges = {tuple(json.load(open(m))["overshoot_bin_edges_in_steps"]) for m in meta}
        if edges != {tuple(OBE)}: fails.append(f"replay bin edges {edges} differ from {OBE}")

    # ---- the feature scan: cells per (patient, admission, channel)
    base = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    seen_adm = {}; n_exc = 0; n_sec = 0; blank_over = 0
    for fp in lk_files:
        rd = csv.DictReader(open(fp))
        if rd.fieldnames is None or "over" not in rd.fieldnames or "D_s" not in rd.fieldnames:
            fails.append(f"{os.path.basename(fp)} has no overshoot column"); continue
        for r in rd:
            key = (r["aMRN"], r["aCSN"])
            if key in seen_adm and seen_adm[key] != fp: fails.append(f"admission in two feature-scan files: duplicate ({os.path.basename(seen_adm[key])}, {os.path.basename(fp)})"); break
            seen_adm[key] = fp
            if r["over"] == "": blank_over += 1; continue
            d = int(r["D_s"]); ob = bisect.bisect_right(OBE, float(r["over"]) / STEP[r["channel"]]) - 1
            c = base[(r["aMRN"], r["aCSN"], r["channel"])][(cell_of(d), ob)]; c[0] += 1; c[1] += d; n_exc += 1; n_sec += d
    if blank_over: fails.append(f"{blank_over} excursion rows without an overshoot")
    if n_exc == 0: fails.append("feature scan holds no excursions: empty input")

    # ---- the replay: offset zero, gap 4
    rep = {}; seen_rp = set(); dup = 0; bad_tot = 0
    for fp in rp_files:
        for r in csv.DictReader(open(fp)):
            k = (r["aMRN"], r["aCSN"], r["channel"], r["offset_step"], r["gap_s"])
            if k in seen_rp: dup += 1; continue
            seen_rp.add(k)
            if int(r["offset_step"]) != 0 or int(r["gap_s"]) != 4: continue
            ent = [tuple(int(x) for x in e.split(":")) for e in r["joint"].split(";")] if r["joint"] else []
            if sum(e[2] for e in ent) != int(r["n_exc"]) or sum(e[3] for e in ent) != int(r["sum_secs"]): bad_tot += 1
            rep[(r["aMRN"], r["aCSN"], r["channel"])] = {(c, ob): [n, s] for c, ob, n, s in ent}
    if dup: fails.append(f"{dup} duplicate replay rows")
    if bad_tot: fails.append(f"{bad_tot} replay rows whose totals disagree with their cells")
    if not rep: fails.append("replay holds no offset-zero rows: empty input")

    # ---- the identity
    keys_b = set(base); keys_r = set(rep)
    only_b = sorted(keys_b - keys_r); only_r = sorted(keys_r - keys_b)
    mism = []; n_cells = 0; n_cells_bad = 0
    for k in sorted(keys_b & keys_r):
        cb = base[k]; cr = rep[k]
        for cell in sorted(set(cb) | set(cr)):
            n_cells += 1
            vb = cb.get(cell, [0, 0]); vr = cr.get(cell, [0, 0])
            if vb != vr: n_cells_bad += 1; mism.append((k, cell, vb, vr))
    if only_b: fails.append(f"{len(only_b)} admission-channel keys in the feature scan but not the replay")
    if only_r: fails.append(f"{len(only_r)} admission-channel keys in the replay but not the feature scan")
    if n_cells_bad: fails.append(f"{n_cells_bad} of {n_cells} cells disagree in count or seconds")
    log = a.mismatch_log or os.path.join(W, "diagnostics", f"JOINT_IDENTITY_{tag}_mismatches.csv")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["aMRN", "aCSN", "channel", "cell", "obin", "feature_n", "feature_secs", "replay_n", "replay_secs"])
        for (m, c, ch), (cell, ob), vb, vr in mism: w.writerow([m, c, ch, cell, ob, vb[0], vb[1], vr[0], vr[1]])
        for k in only_b: w.writerow(list(k) + ["", "", "", "", "absent", "absent"])
        for k in only_r: w.writerow(list(k) + ["", "", "absent", "absent", "", ""])
    summary = {"tag": tag, "feature_stripes": n_lk, "replay_stripes": n_rp, "admissions_intended": sum(len(x) for x in exp_lk),
               "admissions_with_excursions": len({(k[0], k[1]) for k in keys_b}), "admission_channel_keys": len(keys_b),
               "excursions": n_exc, "excursion_seconds": n_sec, "cells_compared": n_cells, "cells_disagreeing": n_cells_bad,
               "keys_only_in_feature_scan": len(only_b), "keys_only_in_replay": len(only_r), "mismatch_log": log}
    return finish(a, tag, fails, notes, summary)


def finish(a, tag, fails, notes, summary):
    out = a.out or os.path.join(a.work, "diagnostics", f"JOINT_IDENTITY_{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    verdict = "IDENTITY HOLDS" if not fails else "IDENTITY FAILS"
    rec = {"verdict": verdict, "failures": fails, "summary": summary}
    json.dump(rec, open(out, "w"), indent=1)
    if summary:
        print(f"{tag}: {summary['admissions_with_excursions']} admissions with excursions of {summary['admissions_intended']} intended, "
              f"{summary['excursions']} excursions, {summary['excursion_seconds']} s, {summary['cells_compared']} cells compared, "
              f"{summary['cells_disagreeing']} disagree")
    for f in fails: print("  -", f)
    print(f"{verdict} ({tag})")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
