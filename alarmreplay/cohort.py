"""Optional restriction of the analysed admissions to a named subset.

The primary analyses use every admission their scan rows contain. A secondary arm may restrict
them to a subset named in a file -- the diagnosis-derived oncology arm of the supplement is the
one this exists for. The subset is a CSV with an `aCSN` column; anything else in the file is
ignored, so the cohort table itself can be passed after filtering.

Two environment variables drive it, both unset in the primary analyses:

    ALARM_COHORT_SUBSET  path of the subset file
    ALARM_COHORT_LABEL   how the arm is named in the output record

`restriction()` returns a `(keep, record)` pair: `keep` is a predicate on a row dictionary that
is `True` for every row when no subset is set, and `record` is the description the analysis writes
into its output so that an arm can never be mistaken for the primary one.
"""
import csv
import os

__all__ = ["restriction", "load_subset"]


def load_subset(path):
    """The admission identifiers in a subset file, as a set of strings."""
    with open(path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "aCSN" not in reader.fieldnames:
            raise SystemExit(f"{path}: no aCSN column; a cohort subset names admissions by aCSN")
        keep = {r["aCSN"].strip() for r in reader if r["aCSN"].strip()}
    if not keep:
        raise SystemExit(f"{path} names no admissions")
    return keep


def restriction(path=None, label=None):
    """(predicate, record) for the cohort restriction in force.

    With no subset the predicate admits every row and the record says so, so the same code path
    runs in the primary analysis and in a restricted arm.
    """
    path = os.environ.get("ALARM_COHORT_SUBSET", "") if path is None else path
    if not path:
        return (lambda row: True), {"restricted": False, "cohort": "all admissions in the scan rows"}
    keep = load_subset(path)
    label = label or os.environ.get("ALARM_COHORT_LABEL", "") or os.path.basename(path)
    record = {"restricted": True, "cohort": label, "subset_file": path,
              "n_admissions_named": len(keep)}
    return (lambda row: row["aCSN"] in keep), record
