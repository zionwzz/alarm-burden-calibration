#!/usr/bin/env python3
"""Remove the duplicated admission blocks a stopped-and-resumed scan can leave in its row files.

Every scanner writes an admission's rows as one contiguous block and records its position a
few admissions later. A process stopped after a block had reached the disk but before the
position was recorded scans that admission again on resume and appends a second block
(`alarmreplay.scan_common.Resumable` prevents this by truncating to the recorded length; this
tool handles row files written without it). The second block is the one the recorded position
covers, so for every admission that appears in more than one contiguous block of a file, this
script keeps the last block and drops the earlier ones. It rewrites the file in place through a
temporary copy, keeps a `.bak` of the file beside it, and reports what it removed. It never
touches a file in which no admission repeats.

usage: dedupe_scan_rows.py <work tree> [--apply]
       (without --apply it only reports)
"""
import csv, glob, json, os, sys

PATTERNS = ["alarm_epidemiology/a1_rows_*.csv", "delay_model/a2_*_rows_*.csv", "sensitivity/s2_rows_*.csv",
            "policy/lk_rows_*.csv", "policy/lk_tot_*.csv", "policy/lk2_rows_*.csv", "policy/lk2_tot_*.csv", "policy/lk2_chg_*.csv",
            "policy/pd_rows_*.csv", "policy/pd2_rows_*.csv", "policy/pd3_rows_*.csv"]


def blocks(rows):
    """Contiguous blocks of rows sharing (aMRN, aCSN): list of (key, start, end)."""
    out = []; cur = None; start = 0
    for i, r in enumerate(rows):
        k = (r[0], r[1])
        if k != cur:
            if cur is not None: out.append((cur, start, i))
            cur = k; start = i
    if cur is not None: out.append((cur, start, len(rows)))
    return out


def main():
    if len(sys.argv) < 2: raise SystemExit(__doc__)
    W = sys.argv[1]; apply = "--apply" in sys.argv
    report = {"files": {}, "applied": apply}
    total_removed = 0
    for pat in PATTERNS:
        for fp in sorted(glob.glob(os.path.join(W, pat))):
            with open(fp, newline="") as f:
                rd = csv.reader(f); hdr = next(rd, None)
                if hdr is None: continue
                rows = list(rd)
            bl = blocks(rows)
            last = {}
            for idx, (k, s, e) in enumerate(bl): last[k] = idx
            drop = [(k, s, e) for idx, (k, s, e) in enumerate(bl) if last[k] != idx]
            if not drop: continue
            keep_mask = [True] * len(rows)
            for k, s, e in drop:
                for i in range(s, e): keep_mask[i] = False
            removed = sum(1 for m in keep_mask if not m); total_removed += removed
            report["files"][os.path.relpath(fp, W)] = {"admissions_repeated": len({k for k, _, _ in drop}), "rows_removed": removed, "rows_kept": len(rows) - removed}
            if apply:
                bak = fp + ".bak"
                if not os.path.exists(bak): os.replace(fp, bak)
                with open(fp + ".tmp", "w", newline="") as f:
                    w = csv.writer(f); w.writerow(hdr); w.writerows(r for r, m in zip(rows, keep_mask) if m)
                os.replace(fp + ".tmp", fp)
    report["rows_removed_total"] = total_removed
    out = os.path.join(W, "diagnostics", "DEDUPE_REPORT.json"); os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=1)
    for fp, r in report["files"].items(): print(f"  {fp}: {r['admissions_repeated']} admission(s) repeated, {r['rows_removed']} rows {'removed' if apply else 'would be removed'}")
    print(f"{'applied' if apply else 'report only'}: {total_removed} rows in {len(report['files'])} file(s); written {out}")


if __name__ == "__main__":
    main()
