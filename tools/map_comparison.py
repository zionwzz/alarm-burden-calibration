#!/usr/bin/env python3
"""The duration-by-overshoot grid under the prespecified (tuning-subsample) cell map beside the
same grid under a map selected in sample: PRESPECIFIED_MAP_COMPARISON.txt.

Both inputs are LINK_ANALYSIS.json files written by analysis/link_analysis.py -- the primary arm
(ALARM_FEATURE_MAP set to the tuning-subsample map) and the sensitivity arm (no map given, so the
script selects one on the analysis sample, in a working tree of its own).
Only aggregates are read, so the output can leave the credentialed environment.

usage: map_comparison.py <LINK_ANALYSIS.json, in-sample map> <LINK_ANALYSIS.json, prespecified map> [out]
"""
import json, sys

CH = [("ecgResp-numLimit-respRate-low", "RR low"), ("spO2-numLimit-satO2-low", "SpO2 low"),
      ("ecg-numLimit-heartRate-high", "HR high"), ("ecg-numLimit-heartRate-low", "HR low")]
SHIFTS = ["-1", "1", "2", "3"]


def main(is_path, pre_path, out=None):
    IS = json.load(open(is_path)); PRE = json.load(open(pre_path))
    if PRE[CH[0][0]]["transport_feature"].get("map_prespecified") is not True: raise SystemExit(f"{pre_path}: not the prespecified-map arm")
    if IS[CH[0][0]]["transport_feature"].get("map_prespecified") is not False: raise SystemExit(f"{is_path}: not the in-sample arm")
    L = ["FULL GRID: duration-by-overshoot transport, in-sample map (IS) vs prespecified tuning-split map (PRE)",
         "Source: two LINK_ANALYSIS.json files written by analysis/link_analysis.py, one per arm (tools/map_comparison.py)",
         "Duration-transport quantities (kappa, corrected %) are IDENTICAL between the two maps by construction:",
         "the duration-only estimator does not use the joint cell map.", "",
         f"{'Channel':<10} {'shift':>5} | {'kz_IS':>7} {'CI_IS':>18} {'corrz_IS':>9} | {'kz_PRE':>7} {'CI_PRE':>18} {'corrz_PRE':>9} | {'d_kz':>7} {'d_corr':>7}",
         "-" * 113]
    mk = mc = 0.0; mk1 = mc1 = 0.0
    for ch, lab in CH:
        for s in SHIFTS:
            a = IS[ch]["transport_feature"]["by_offset"][s]; b = PRE[ch]["transport_feature"]["by_offset"][s]
            dk = b["kappa_z"] - a["kappa_z"]; dc = b["corrected_reduction_z_pct"] - a["corrected_reduction_z_pct"]
            mk = max(mk, abs(dk)); mc = max(mc, abs(dc))
            if s == "1": mk1 = max(mk1, abs(dk)); mc1 = max(mc1, abs(dc))
            L.append(f"{lab:<10} {s:>5} | {a['kappa_z']:>7.4f} {'(' + format(a['CI'][0], '.3f') + '-' + format(a['CI'][1], '.3f') + ')':>18} {a['corrected_reduction_z_pct']:>9.2f} | "
                     f"{b['kappa_z']:>7.4f} {'(' + format(b['CI'][0], '.3f') + '-' + format(b['CI'][1], '.3f') + ')':>18} {b['corrected_reduction_z_pct']:>9.2f} | {dk:>+7.4f} {dc:>+7.2f}")
        L.append("")
    L.append("Coarsened bulk cells retained/coarsened out of 384 joint cells:")
    for ch, lab in CH:
        L.append(f"  {lab:<9}  IS coarsened={IS[ch]['transport_feature']['n_bulk_cells_fallback']:>4}   PRE coarsened={PRE[ch]['transport_feature']['n_bulk_cells_fallback']:>4}   "
                 f"(cells with their own reward: IS {IS[ch]['transport_feature']['map_cells_retained']}, PRE {PRE[ch]['transport_feature']['map_cells_retained']})")
    L += ["", "MAXIMUM MOVEMENT: |d_kz| and |d_corr| across the whole grid:",
          f"  max |change in kappa_z| = {mk:.4f}", f"  max |change in corrected reduction| = {mc:.2f} percentage points",
          f"  at the +1 step alone: {mk1:.4f} and {mc1:.2f} percentage points", "",
          "Duration-transport (unchanged by map choice), for reference, at +1 step:"]
    for ch, lab in CH:
        k = PRE[ch]["kappa"]["1"]
        L.append(f"  {lab:<9}  replayed={k['replayed_reduction_pct']:>7.2f}  kappa_d={k['kappa']:.4f} CI=({k['CI'][0]:.4f}-{k['CI'][1]:.4f})  "
                 f"corrected_d={k['corrected_reduction_pct']:.2f} CI=({k['corrected_CI'][0]:.2f}-{k['corrected_CI'][1]:.2f})")
    text = "\n".join(L) + "\n"
    if out:
        with open(out, "w") as fh: fh.write(text)
    print(text, end="")


if __name__ == "__main__":
    if len(sys.argv) < 3: raise SystemExit(__doc__)
    main(*sys.argv[1:4])
