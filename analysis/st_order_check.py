#!/usr/bin/env python3
"""The stochastic-order premise of the replay-direction argument, checked on the replayed durations.

The argument needs the length-biased duration law at the one-step less stringent limit to lie
below the one at the limit in force, D~(u1) <=_st D~(u0): at every duration d, the share of
replayed seconds in excursions no longer than d must be at least as large after the shift as
before. The check is made on the replayed durations the estimator itself uses -- the marginal
replay of the every-fourth fitting subsample (pd2 rows, gap 4 s) at offsets 0 and +1, on the
estimator's own cells: sixty 2-second cells weighted by count times duration and four tail
bins weighted by their recorded seconds. The reported value is the largest violation over the
cell boundaries, max_d [F~_0(d) - F~_1(d)]^+, so zero means the premise holds on the grid and
a positive value is the size of the largest crossing.

A second reading -- the whole fitting split's marginal replay with every excursion of 120 s or
more pooled at a representative 150 s -- is computed beside it, so that both are on the record.

Writes policy/ST_ORDER_CHECK.json.
"""
from alarmreplay.paths import WORK as R
import csv, glob, json, os, sys, collections

NB = 60; NT = 4; TAIL_REP = 150.0
CH = {"ecgResp-numLimit-respRate-low": "RR low", "spO2-numLimit-satO2-low": "SpO2 low",
      "ecg-numLimit-heartRate-high": "HR high", "ecg-numLimit-heartRate-low": "HR low"}


def cdf_from_mass(mass):
    tot = sum(mass); out = []; acc = 0.0
    for m in mass:
        acc += m; out.append(acc / tot if tot else 0.0)
    return out, tot


def subsample_cells(pattern):
    """pd2 rows: (channel, offset) -> length-biased mass on 60 cells + 4 tail bins."""
    H = collections.defaultdict(lambda: [0.0] * (NB + NT)); seen = set(); n = 0
    for fp in sorted(glob.glob(pattern)):
        for r in csv.DictReader(open(fp)):
            k = (r["aMRN"], r["aCSN"], r["channel"], r["offset_step"], r["gap_s"])
            if k in seen or int(r["gap_s"]) != 4 or r["offset_step"] not in ("0", "1"): continue
            seen.add(k); n += 1
            h = [int(x) for x in r["dwell_hist"].split("|")]; ts = [int(x) for x in r["tail_secs"].split("|")]
            acc = H[(r["channel"], r["offset_step"])]
            for i, c in enumerate(h): acc[i] += c * 2.0 * i          # cell i holds excursions of 2i seconds; cell 0 is empty
            for i, s in enumerate(ts): acc[NB + i] += s
    return H, n


def full_split_61(pattern):
    """pd rows (whole fitting split): (channel, offset) -> mass on 60 cells + one pooled tail at 150 s."""
    H = collections.defaultdict(lambda: [0.0] * (NB + 1)); seen = set(); n = 0
    for fp in sorted(glob.glob(pattern)):
        for r in csv.DictReader(open(fp)):
            k = (r["aMRN"], r["aCSN"], r["channel"], r["offset_step"])
            if k in seen or int(r["gap_s"]) != 4 or r["offset_step"] not in ("0", "1"): continue
            seen.add(k); n += 1
            h = [int(x) for x in r["dwell_hist"].split("|")]
            acc = H[(r["channel"], r["offset_step"])]
            for i, c in enumerate(h): acc[i] += c * 2.0 * i
            acc[NB] += int(r["n_over120"]) * TAIL_REP
    return H, n


def violation(H, ch):
    if (ch, "0") not in H or (ch, "1") not in H: return None
    F0, t0 = cdf_from_mass(H[(ch, "0")][1:]); F1, t1 = cdf_from_mass(H[(ch, "1")][1:])
    return max(0.0, max(a - b for a, b in zip(F0, F1))), t0, t1


def main(pd2_pattern=None, pd_pattern=None, out=None):
    pd2_pattern = pd2_pattern or f"{R}/policy/pd2_rows_[0-9]*.csv"; pd_pattern = pd_pattern or f"{R}/policy/pd_rows_[0-9]*.csv"
    H2, n2 = subsample_cells(pd2_pattern); H1, n1 = full_split_61(pd_pattern)
    if n2 == 0: raise SystemExit(f"no subsample replay rows under {pd2_pattern}")
    res = {}
    for ch, label in CH.items():
        v2 = violation(H2, ch)
        if v2 is None: raise SystemExit(f"{ch}: offsets 0 and +1 needed in the subsample replay")
        rec = {"max_st_violation_lengthbiased_plus1": round(v2[0], 4), "definition": "every-fourth fitting subsample, 60 cells + 4 tail bins by seconds",
               "length_biased_mass_offset0": v2[1], "length_biased_mass_offset1": v2[2]}
        v1 = violation(H1, ch) if n1 else None
        rec["full_split_61cell_variant"] = ({"max_st_violation_lengthbiased_plus1": round(v1[0], 4),
                                             "definition": "whole fitting split, 60 cells + one tail at 150 s"}
                                            if v1 is not None else
                                            {"max_st_violation_lengthbiased_plus1": None,
                                             "note": "not computed: the whole-split marginal replay of this run covers the recorded limit only (offset 0), and the variant needs the +1 step as well"})
        res[label] = rec
    out = out or f"{R}/policy/ST_ORDER_CHECK.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    for label, rec in res.items():
        alt = rec["full_split_61cell_variant"]["max_st_violation_lengthbiased_plus1"]
        print(f"{label:<9} violation {rec['max_st_violation_lengthbiased_plus1']:.4f} (subsample cells)   full-split 61-cell variant {'not computed' if alt is None else format(alt, '.4f')}")
    print("written", out)


if __name__ == "__main__":
    main(*(sys.argv[1:4]))
