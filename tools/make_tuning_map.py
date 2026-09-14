#!/usr/bin/env python3
"""Select the duration-by-overshoot cell map on the independent tuning subsample.

WHAT A CELL MAP IS. The duration-by-overshoot estimator splits each duration cell by how far the
excursion went beyond the limit. A joint (duration, overshoot) cell with too few baseline excursions
cannot support its own reward, so it is *coarsened*: its reward is replaced by the duration-marginal
reward. The cell map is the boolean array that says which joint cells keep their own reward. It is
one array per channel, of shape (number of duration cells) x (number of overshoot bins).

WHY IT IS BUILT HERE AND NOT INSIDE THE ANALYSIS. The fixed-map inference assumes the cell map --- the
retained joint cells and the coarsened complement together --- is fixed independently of the sample
the estimator is computed on. Selecting it on the analysis sample and then holding it fixed inside
the bootstrap does not satisfy that. The design record carries an independent tuning subsample, a
quarter of the patients, disjoint from the subsample the estimates are computed on. This script builds the map from that subsample's excursions alone, using exactly the cell and
overshoot binning `analysis/link_analysis.py` uses, and writes it where `ALARM_FEATURE_MAP` can pick
it up. The map then enters the analysis split unchanged, is used for every candidate limit, and is
held fixed in every bootstrap replicate, which is what the fixed-map inference assumes.

**The results the paper reports as primary are computed under this map.** Running the analysis
without it selects a map in sample instead, which is the sensitivity arm, and reproduces different
duration-by-overshoot numbers. `PRESPECIFIED_MAP_COMPARISON.txt` in the repository root carries the
two grids side by side.

USAGE.

    ALARM_WORK=<work> python3 tools/make_tuning_map.py
    ALARM_FEATURE_MAP=<work>/design/FEATURE_MAP_TUNE.json ALARM_WORK=<work> \
        python3 analysis/link_analysis.py

or, equivalently, `ALARM_WORK=<work> make analysis`, which runs `make tuningmap` first and passes the
map to the two scripts that read it (`analysis/link_analysis.py` and
`analysis/transport_diagnostics.py`). README.md, "The duration-by-overshoot cell map", has the whole
sequence.

INPUT. `$ALARM_WORK/policy/lk2_rows_<tag>_*.csv`, the per-excursion linked-feature scan rows for the
tuning subsample, written by `scan/scan_linkage_features.py`. They are part of the credentialed
release; this script cannot be run without them. Only two columns are read:
`D_s`, the excursion duration in seconds, and `over`, how far the value went beyond the limit in the
channel's own units (empty where no overshoot was recorded, which is treated as the lowest bin).

OUTPUT. `$ALARM_WORK/design/FEATURE_MAP_TUNE.json`: a JSON object keyed by channel, each value the
boolean cell map as a nested list, rows indexed by duration cell and columns by overshoot bin, in the
shape `analysis/link_analysis.py` asserts on load. The `design/` directory is created if it is not
there. Running again overwrites the file; the selection is deterministic, so a repeat run on the same scan
rows writes the same map.

ENVIRONMENT.
  ALARM_WORK         the working tree (`alarmreplay/paths.py`); default `<repo>/work`
  NMIN               minimum baseline excursions before a joint cell keeps its own reward, default 5.
                     `analysis/link_analysis.py` and `analysis/transport_diagnostics.py` read the
                     same variable, and the map is only coherent with the analysis if all three see
                     the same value.
  ALARM_TUNING_TAG   the scan-row tag identifying the tuning subsample, default `TUNE2p0`. The tag
                     encodes which subsample and which merge gap the rows were scanned under; change
                     it only to point at a differently tagged tuning scan.
"""
import collections
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alarmreplay.linked_reward import CellMap, feature_support_map      # noqa: E402
from alarmreplay.paths import WORK as R, ensure                          # noqa: E402

# Every constant below must match analysis/link_analysis.py exactly. They are the definition of a
# cell, and a map built on a different grid from the one the estimator uses is not a map of anything:
# link_analysis.py asserts the loaded array's shape, which catches a changed grid size but not a
# changed bin edge. Change one here and you must change the same one there, in the same commit.
O = f"{R}/policy"
CH = ["ecgResp-numLimit-respRate-low", "spO2-numLimit-satO2-low",
      "ecg-numLimit-heartRate-high", "ecg-numLimit-heartRate-low"]
NO = 6                          # overshoot bins per duration cell
OBE = [0, 1, 2, 4, 8, 16]       # their lower edges, in steps of the channel's own limit grid
# One step of the limit grid, per channel: 5 beats/min on the two heart-rate channels, 1 breath/min
# on respiration rate, 1 percentage point on SpO2. `over` is divided by this before binning, so an
# overshoot bin means the same number of limit steps on every channel.
STEP = {"ecg-numLimit-heartRate-high": 5.0, "ecg-numLimit-heartRate-low": 5.0,
        "spO2-numLimit-satO2-low": 1.0, "ecgResp-numLimit-respRate-low": 1.0}
# Durations: 60 exact 2-second cells covering 0-120 s, then four pooled tail bins above 120 s.
TAILB = [120, 180, 300, 600]
CMAP = CellMap(width=2.0, n_exact=60, tail_edges=tuple(float(x) for x in TAILB),
               tail_midpoints=(150.0, 240.0, 450.0, 900.0))
NC = CMAP.n_cells               # 64 duration cells; 64 x 6 = 384 joint cells per channel
NMIN = int(os.environ.get("NMIN", "5"))
TAG = os.environ.get("ALARM_TUNING_TAG", "TUNE2p0")


def main():
    # The tuning subsample's excursions, one row each, across however many scan stripes were run.
    rows = collections.defaultdict(list)
    files = sorted(glob.glob(f"{O}/lk2_rows_{TAG}_*.csv"))
    if not files:
        sys.exit(
            f"no tuning-subsample scan rows found: nothing matches {O}/lk2_rows_{TAG}_*.csv\n"
            f"  ALARM_WORK resolves to {R}; set it to the working tree that holds the scan rows.\n"
            f"  ALARM_TUNING_TAG is {TAG!r}; set it if the tuning scan was written under another tag.\n"
            f"  These rows are written by scan/scan_linkage_features.py from the credentialed\n"
            f"  telemetry release, so this script needs a working tree of a credentialed run.")
    for fp in files:
        for r in csv.DictReader(open(fp)):
            if r["channel"] in STEP:
                rows[r["channel"]].append(r)

    out, report = {}, []
    for ch in CH:
        Rw = rows[ch]
        d = np.array([float(r["D_s"]) for r in Rw])
        # An empty `over` means no overshoot was recorded for that excursion; it is put in the
        # lowest bin, which is what link_analysis.py does with the same column.
        ov = np.array([float(r["over"]) if r["over"] != "" else np.nan for r in Rw])
        c = np.array([int(CMAP.index(x)) for x in d])            # duration cell of each excursion
        ob = np.where(np.isnan(ov), 0,                            # overshoot bin of each excursion
                      np.clip(np.searchsorted(OBE, np.nan_to_num(ov) / STEP[ch], side="right") - 1,
                              0, NO - 1))
        # Baseline excursion counts per joint cell, then the retention rule at NMIN. The rule itself
        # lives in alarmreplay.linked_reward, shared with the estimator and the simulation, so the
        # map is selected by the same function that would have selected it in sample.
        counts = np.zeros((NC, NO))
        np.add.at(counts, (c, ob), 1.0)
        ok = feature_support_map(counts, NMIN)
        out[ch] = ok.tolist()
        report.append((ch, len(Rw), int(ok.sum()), ok.size))

    dest = os.path.join(ensure(f"{R}/design"), "FEATURE_MAP_TUNE.json")
    with open(dest, "w") as f:
        json.dump(out, f)
    # The report is the record of what was selected: how many excursions each channel's map was built
    # on, and how many of its joint cells keep their own reward. The complement is coarsened onto the
    # duration-marginal reward. Compare these counts with the `map_cells_retained` the estimator
    # records to confirm the map that was built is the map that was applied.
    print(f"wrote {dest}  (tuning subsample {TAG}, NMIN={NMIN})")
    for ch, n, k, tot in report:
        print(f"  {ch:34} {n:7d} excursions -> {k:3d}/{tot} joint cells keep their own reward")


if __name__ == "__main__":
    main()
