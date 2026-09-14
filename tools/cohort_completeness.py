#!/usr/bin/env python3
"""Cohort completeness of a scan tree: what each scan was meant to cover, what it processed,
what it wrote, and what it could not read.

Two files agreeing with each other do not show that either covered its cohort, so this report
is built from three independent things: the intended admission list of every job (from the
cohort table and the split assignment, exactly as the scanners build it), the position each
stripe's state file records, and the admissions that actually appear in the output rows -- plus
the read logs the scanners keep of admissions they skipped, and, for the epidemiology scan, its
own marker rows for a missing alarm file or a read error.

An admission can legitimately have no output row: the linkage, replay and delay-model scans
write nothing for an admission whose alarm stream carries none of the four limit alarms, or
whose measurements never cross. Those are counted separately from admissions that were skipped
because a file was missing or unreadable.

Only aggregate counts are written; nothing patient-level leaves the working tree.
"""
import argparse, collections, csv, glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.run_scans import load_cohort, jobs, progress  # noqa: E402


def admissions_in(work, patterns):
    seen = set()
    for pat in patterns:
        for fp in glob.glob(os.path.join(work, pat)):
            if fp.endswith(".json"):
                continue
            with open(fp, newline="") as f:
                rd = csv.DictReader(f)
                for r in rd:
                    if "aMRN" in r: seen.add((r["aMRN"], r["aCSN"]))
    return seen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True); ap.add_argument("--cohort", required=True); ap.add_argument("--stripes", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    W = a.work; N = a.stripes
    rows, splitmap = load_cohort(a.cohort, os.path.join(W, "design", "SPLIT_ASSIGNMENT.csv"))
    J = jobs(rows, splitmap, N)
    report = {"cohort_admissions": len(rows), "splits": {s: sum(1 for m, _ in rows if splitmap.get(m) == s) for s in ("FIT", "TUNE", "TEST")},
              "patients": {s: len({m for m, _ in rows if splitmap.get(m) == s}) for s in ("FIT", "TUNE", "TEST")}, "jobs": []}
    # read logs, keyed by the state-file prefix each scanner derives its log name from
    readlog = collections.defaultdict(lambda: collections.Counter())
    for fp in glob.glob(os.path.join(W, "cache", "*_readlog_*.csv")):
        fam = re.sub(r"_\d+(_\d+)?\.csv$", "", os.path.basename(fp))     # strip the stripe (and stripe-count) suffix
        for r in csv.DictReader(open(fp)):
            readlog[fam][r["reason"].split(":")[0]] += 1
    lines = [f"cohort: {len(rows)} admissions; " + ", ".join(f"{s} {report['splits'][s]} admissions / {report['patients'][s]} patients" for s in ("FIT", "TUNE", "TEST"))]
    lines.append(f"{'job':<12} {'intended':>8} {'processed':>9} {'with rows':>9} {'no rows':>8} {'missing':>8} {'read err':>8}  state")
    for j in J:
        done = progress(W, j, N); processed = sum(done); comp = all(x >= y for x, y in zip(done, j["intended"]))
        outs = [o.replace("{k}", "[0-9]*").replace("{N}", str(N)) for o in j["outputs"]]
        with_rows = admissions_in(W, outs) if not outs[0].endswith(".json") else None
        key = j["state"].split("/")[-1].replace("_state_", "_readlog_").rsplit("_{k}", 1)[0].replace("_{N}", "")
        rl = readlog.get(key, collections.Counter())
        missing = rl.get("missing_file", 0); readerr = sum(v for k, v in rl.items() if k != "missing_file")
        if j["name"] == "a1":
            a1 = collections.Counter()
            for fp in glob.glob(os.path.join(W, "alarm_epidemiology", "a1_rows_*.csv")):
                for r in csv.DictReader(open(fp)):
                    if r["alarm_base"] in ("__NO_ALARM_FILE__", "__READ_ERROR__"): a1[r["alarm_base"]] += 1
            missing = a1["__NO_ALARM_FILE__"]; readerr = a1["__READ_ERROR__"]
        nrows = len(with_rows) if with_rows is not None else None
        norows = (processed - nrows - missing - readerr) if nrows is not None else None
        rec = {"job": j["name"], "intended": j["n"], "processed": processed, "complete": comp, "admissions_with_rows": nrows,
               "admissions_without_rows": norows, "missing_file": missing, "read_error": readerr, "per_stripe_intended": j["intended"], "per_stripe_processed": done}
        report["jobs"].append(rec)
        lines.append(f"{j['name']:<12} {j['n']:>8} {processed:>9} {('' if nrows is None else nrows):>9} {('' if norows is None else norows):>8} {missing:>8} {readerr:>8}  {'complete' if comp else 'INCOMPLETE'}")
    report["all_complete"] = all(r["complete"] for r in report["jobs"])
    out = a.out or os.path.join(W, "diagnostics", "COHORT_COMPLETENESS.json")
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(report, open(out, "w"), indent=1)
    print("\n".join(lines)); print("ALL COMPLETE" if report["all_complete"] else "INCOMPLETE")
    return 0 if report["all_complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
