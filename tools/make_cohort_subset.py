#!/usr/bin/env python3
"""Write the admission subset for a restricted analysis arm, from the cohort table's own labels.

The cohort table carries one row per admission with the diagnosis-derived oncology `tier` the
descriptive sections already use; the cancer group within the strict tier is in the working tree's
cancer-group cache (analysis/cancer_groups.py). This tool selects by either and writes an aCSN
list for `alarmreplay/cohort.py`. Nothing but counts is printed.

usage:
  python3 tools/make_cohort_subset.py --tier strict_onc --out <work>/design/COHORT_strict_onc.csv
  python3 tools/make_cohort_subset.py --group haematological --out <path>
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alarmreplay.paths import WORK as R, COHORT as SAMP  # noqa: E402

TIERS = ["strict_onc", "onc_context_nonstrict", "non_onc"]
GROUPS = ["haematological", "thoracic", "gastrointestinal", "other_solid"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=TIERS, help="diagnosis-derived oncology tier")
    ap.add_argument("--group", choices=GROUPS, help="cancer group within the strict tier")
    ap.add_argument("--cohort", default=SAMP, help="cohort table (default: ALARM_COHORT)")
    ap.add_argument("--out", required=True, help="where to write the subset")
    a = ap.parse_args()
    if bool(a.tier) == bool(a.group):
        raise SystemExit("give exactly one of --tier and --group")

    if a.tier:
        rows = [r for r in csv.DictReader(open(a.cohort)) if r.get("tier") == a.tier]
        keep = sorted({r["aCSN"] for r in rows}, key=int)
        label = f"diagnosis-derived oncology tier: {a.tier}"
    else:
        cache = os.path.join(R, "cache", "cancer_group_by_admission.csv")
        if not os.path.exists(cache):
            raise SystemExit(f"{cache} is missing: run analysis/cancer_groups.py first")
        keep = sorted({r["aCSN"] for r in csv.DictReader(open(cache))
                       if r["cancer_group"] == a.group}, key=int)
        label = f"cancer group within the strict oncology tier: {a.group}"
    if not keep:
        raise SystemExit(f"no admissions match; the cohort table has no rows for that selection")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aCSN"])
        for c in keep:
            w.writerow([c])
    print(f"{label}: {len(keep)} admissions written to {os.path.basename(a.out)}")


if __name__ == "__main__":
    main()
