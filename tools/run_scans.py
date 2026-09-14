#!/usr/bin/env python3
"""Run the eight scan families under the shared conventions into a fresh working tree.

The runner is a plan by default: it lists every job it would run -- scanner, sample, stripes,
the number of admissions each stripe should process -- and the paths it would read and
write, and stops. With ``--execute`` it runs one round at a time: the first job that is not
yet complete gets one stripe process per stripe for ``--budget`` seconds, in parallel, and the
runner returns when they do, so that an invocation always fits inside a bounded call. Every
scanner is resumable through its state file, so the runner is simply invoked again until every
job reports complete; ``--status`` prints the position without running anything.

``--work`` must be empty or a tree this runner created (``RUN_RECORD.json`` present); the
design files -- the split assignment and the design record -- are copied in from ``--design-dir``
unchanged, and a recorded delay-model selection, when one is to be kept rather than re-selected,
from ``--selected-model``.

Jobs, in order (later jobs read nothing from earlier ones except s2, which takes the cohort's
modal limits from the epidemiology rows):

  a1        scan_alarm_epidemiology.py      whole cohort
  a2_FIT    scan_delay_model.py FIT         fitting split
  a2_TUNE   scan_delay_model.py TUNE        tuning split (the selection's second check)
  a2_TEST   scan_delay_model.py TEST        held-out split (read once, for the adjudication)
  c1        scan_onset_lead.py              fitting split
  lk        scan_linkage.py                 every-fourth fitting admission, phase 0
  lk2_*     scan_linkage_features.py        FIT phase 0, FIT phase 2, TUNE phase 0
  pd2       scan_counterfactual_limits.py   every-fourth fitting admission, five offsets
  pd_full   the same at the recorded limit over the whole fitting split (representativeness)
  pd3_*     scan_counterfactual_joint.py    FIT phase 0, FIT phase 2, TUNE phase 0
  s2        scan_measurement_conventions.py held-out split, eleven conventions
"""
import argparse, csv, datetime, json, os, shutil, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN = os.path.join(REPO, "scan")


def load_cohort(cohort, split_csv):
    splitmap = {r["aMRN"]: r["split"] for r in csv.DictReader(open(split_csv))}
    rows = [(r["aMRN"], r["aCSN"]) for r in csv.DictReader(open(cohort))]
    return rows, splitmap


def sample(rows, splitmap, split=None, sub=1, phase=0):
    adms = rows if split is None else [a for a in rows if splitmap.get(a[0]) == split]
    return [a for i, a in enumerate(adms) if i % sub == phase]


def jobs(rows, splitmap, nstr):
    """The job list with each stripe's intended admission count."""
    def stripes(adms):
        return [len([a for i, a in enumerate(adms) if i % nstr == k]) for k in range(nstr)]
    J = []
    def add(name, script, args, env, state, adms, outputs):
        J.append(dict(name=name, script=script, args=args, env=env, state=state, intended=stripes(adms), n=len(adms), outputs=outputs))
    add("a1", "scan_alarm_epidemiology.py", ["{k}", "{N}", "{B}"], {}, "cache/a1_state_{k}.json", sample(rows, splitmap), ["alarm_epidemiology/a1_rows_{k}.csv"])
    add("a2_FIT", "scan_delay_model.py", ["FIT", "{k}", "{N}", "{B}"], {}, "cache/a2_FIT_state_{k}.json", sample(rows, splitmap, "FIT"), ["delay_model/a2_FIT_rows_{k}.csv"])
    add("a2_TUNE", "scan_delay_model.py", ["TUNE", "{k}", "{N}", "{B}"], {}, "cache/a2_TUNE_state_{k}.json", sample(rows, splitmap, "TUNE"), ["delay_model/a2_TUNE_rows_{k}.csv"])
    add("a2_TEST", "scan_delay_model.py", ["TEST", "{k}", "{N}", "{B}"], {}, "cache/a2_TEST_state_{k}.json", sample(rows, splitmap, "TEST"), ["delay_model/a2_TEST_rows_{k}.csv"])
    add("c1", "scan_onset_lead.py", ["{k}", "{N}", "{B}"], {}, "cache/c1_state_{k}_{N}.json", sample(rows, splitmap, "FIT"), ["comparison/c1_onset_hist_{k}_{N}.json"])
    add("lk", "scan_linkage.py", ["{k}", "{N}", "{B}"], {}, "cache/lk_state_{k}.json", sample(rows, splitmap, "FIT", 4, 0), ["policy/lk_rows_{k}.csv", "policy/lk_tot_{k}.csv"])
    add("lk2_FIT4p0", "scan_linkage_features.py", ["FIT", "4", "0", "{k}", "{N}", "{B}"], {}, "cache/lk2_state_FIT4p0_{k}.json", sample(rows, splitmap, "FIT", 4, 0), ["policy/lk2_rows_FIT4p0_{k}.csv", "policy/lk2_tot_FIT4p0_{k}.csv", "policy/lk2_chg_FIT4p0_{k}.csv"])
    add("lk2_FIT4p2", "scan_linkage_features.py", ["FIT", "4", "2", "{k}", "{N}", "{B}"], {}, "cache/lk2_state_FIT4p2_{k}.json", sample(rows, splitmap, "FIT", 4, 2), ["policy/lk2_rows_FIT4p2_{k}.csv", "policy/lk2_tot_FIT4p2_{k}.csv", "policy/lk2_chg_FIT4p2_{k}.csv"])
    add("lk2_TUNE2p0", "scan_linkage_features.py", ["TUNE", "2", "0", "{k}", "{N}", "{B}"], {}, "cache/lk2_state_TUNE2p0_{k}.json", sample(rows, splitmap, "TUNE", 2, 0), ["policy/lk2_rows_TUNE2p0_{k}.csv", "policy/lk2_tot_TUNE2p0_{k}.csv", "policy/lk2_chg_TUNE2p0_{k}.csv"])
    add("pd2", "scan_counterfactual_limits.py", ["{k}", "{N}", "{B}"], {}, "cache/pd2_state_{k}.json", sample(rows, splitmap, "FIT", 4, 0), ["policy/pd2_rows_{k}.csv"])
    add("pd_full", "scan_counterfactual_limits.py", ["{k}", "{N}", "{B}"], {"PD_FULL": "1"}, "cache/pd_state_{k}.json", sample(rows, splitmap, "FIT"), ["policy/pd_rows_{k}.csv"])
    add("pd3_FIT4p0", "scan_counterfactual_joint.py", ["{k}", "{N}", "{B}"], {}, "cache/pd3_state_{k}.json", sample(rows, splitmap, "FIT", 4, 0), ["policy/pd3_rows_{k}.csv"])
    add("pd3_FIT4p2", "scan_counterfactual_joint.py", ["{k}", "{N}", "{B}"], {"PD3_PHASE": "2"}, "cache/pd3_state_p2_{k}.json", sample(rows, splitmap, "FIT", 4, 2), ["policy/pd3_rows_p2_{k}.csv"])
    add("pd3_TUNE2p0", "scan_counterfactual_joint.py", ["{k}", "{N}", "{B}"], {"PD3_SPLIT": "TUNE", "PD3_SUB": "2"}, "cache/pd3_state_TUNE2p0_{k}.json", sample(rows, splitmap, "TUNE", 2, 0), ["policy/pd3_rows_TUNE2p0_{k}.csv"])
    add("s2", "scan_measurement_conventions.py", ["{k}", "{N}", "{B}"], {}, "cache/s2_state_{k}.json", sample(rows, splitmap, "TEST"), ["sensitivity/s2_rows_{k}.csv"])
    return J


def progress(work, job, nstr):
    done = []
    for k in range(nstr):
        st = os.path.join(work, job["state"].format(k=k, N=nstr))
        i = json.load(open(st))["i"] if os.path.exists(st) else 0
        done.append(min(i, job["intended"][k]))
    return done


def status(work, J, nstr):
    lines = []; all_done = True
    for j in J:
        d = progress(work, j, nstr); comp = all(a >= b for a, b in zip(d, j["intended"]))
        all_done &= comp
        lines.append(f"  {j['name']:<12} {'complete' if comp else 'pending ':<9} {sum(d):>6}/{j['n']:<6} admissions  stripes {'/'.join(str(x) for x in d)} of {'/'.join(str(x) for x in j['intended'])}")
    return all_done, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="the fresh working tree to write")
    ap.add_argument("--telemetry", required=True); ap.add_argument("--cohort", required=True)
    ap.add_argument("--design-dir", required=True, help="directory holding SPLIT_ASSIGNMENT.csv and DESIGN.json")
    ap.add_argument("--selected-model", default=None, help="a recorded A2_SELECTED_MODEL.json to keep rather than re-select")
    ap.add_argument("--stripes", type=int, default=4); ap.add_argument("--budget", type=float, default=3600.0, help="seconds per stripe process per round")
    ap.add_argument("--wall", type=float, default=float("inf"), help="seconds this invocation may use in total (default: until complete)")
    ap.add_argument("--only", default=None, help="comma-separated job names to restrict to")
    ap.add_argument("--execute", action="store_true"); ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    W = os.path.abspath(a.work); N = a.stripes
    split_csv = os.path.join(a.design_dir, "SPLIT_ASSIGNMENT.csv"); design_json = os.path.join(a.design_dir, "DESIGN.json")
    for p in (a.telemetry, a.cohort, split_csv, design_json):
        if not os.path.exists(p): raise SystemExit(f"missing input: {p}")
    rows, splitmap = load_cohort(a.cohort, split_csv)
    J = jobs(rows, splitmap, N)
    if a.only: J = [j for j in J if j["name"] in a.only.split(",")]
    run_record = os.path.join(W, "RUN_RECORD.json")

    if not a.execute and not a.status:
        counts = ", ".join(f"{s}={sum(1 for m, _ in rows if splitmap.get(m) == s)}" for s in ("FIT", "TUNE", "TEST"))
        print(f"PLAN (nothing run). telemetry={a.telemetry}\n  cohort={a.cohort} ({len(rows)} admissions; {counts})\n"
              f"  design={a.design_dir}\n  selected model={a.selected_model}\n"
              f"  work={W}\n  stripes={N} budget={a.budget:.0f}s per round, wall={'until complete' if a.wall == float('inf') else f'{a.wall:.0f}s'} per invocation")
        for j in J:
            print(f"  {j['name']:<12} {j['script']:<34} args={' '.join(j['args'])} env={j['env'] or '{}'}\n"
                  f"               admissions={j['n']} per stripe={j['intended']}  writes {', '.join(j['outputs'])}")
        return 0

    # ---- the tree: fresh, or one this runner made
    if os.path.exists(run_record):
        M = json.load(open(run_record))
        if os.path.abspath(M["telemetry"]) != os.path.abspath(a.telemetry) or os.path.abspath(M["cohort"]) != os.path.abspath(a.cohort) or M["stripes"] != N:
            raise SystemExit("this tree was started with different inputs or a different stripe count; use a new --work")
    else:
        if os.path.exists(W) and any(os.scandir(W)):
            raise SystemExit(f"{W} exists and is not empty and has no RUN_RECORD.json: refusing to write into it")
        for sub in ("design", "alarm_epidemiology", "delay_model", "comparison", "sensitivity", "policy", "diagnostics", "cache"):
            os.makedirs(os.path.join(W, sub), exist_ok=True)
        shutil.copy(split_csv, os.path.join(W, "design", "SPLIT_ASSIGNMENT.csv")); shutil.copy(design_json, os.path.join(W, "design", "DESIGN.json"))
        if a.selected_model: shutil.copy(a.selected_model, os.path.join(W, "delay_model", "A2_SELECTED_MODEL.json"))
        try: commit = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        except Exception: commit = ""
        M = {"created": datetime.datetime.utcnow().isoformat() + "Z", "telemetry": a.telemetry, "cohort": a.cohort, "design_dir": a.design_dir,
             "selected_model": a.selected_model, "stripes": N, "repository_commit": commit,
             "jobs": [{k: v for k, v in j.items() if k != "env"} | {"env": j["env"]} for j in J]}
        json.dump(M, open(run_record, "w"), indent=1)
    # the job list on record is always the full current plan, whatever subset an invocation ran
    if not a.only:
        M["jobs"] = [{k: v for k, v in j.items()} for j in J]; json.dump(M, open(run_record, "w"), indent=1)
    all_done, lines = status(W, J, N)
    if a.status:
        print("\n".join(lines)); print("ALL COMPLETE" if all_done else "INCOMPLETE"); return 0 if all_done else 2

    # ---- execute rounds until the wall is spent or everything is complete
    t0 = time.time(); log = open(os.path.join(W, "RUN_LOG.jsonl"), "a")
    env = dict(os.environ, ALARM_WORK=W, ALARM_TELEMETRY=a.telemetry, ALARM_COHORT=a.cohort, PYTHONPATH=REPO)
    # A small scheduler: up to N stripe processes at a time, refilled as soon as one exits, so
    # the tail of one job and the head of the next share the cores. Pending (job, stripe) pairs
    # are taken in job order; s2 is last and needs a1 complete, which the order guarantees.
    def pending_pairs():
        pend = [(j, k) for j in J for k in range(N) if progress(W, j, N)[k] < j["intended"][k]]
        if any(j["name"] == "a1" for j, _ in pend):
            pend = [(j, k) for j, k in pend if j["name"] != "s2"]
        return pend

    def launch(j, k, B):
        args = [x.format(k=k, N=N, B=f"{B:.0f}") for x in j["args"]]
        return subprocess.Popen([sys.executable, os.path.join(SCAN, j["script"]), *args], env=dict(env, **j["env"]),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=REPO)

    def record(j, k, p):
        out, err = p.communicate()
        rec = {"t": datetime.datetime.utcnow().isoformat() + "Z", "job": j["name"], "stripe": k, "rc": p.returncode,
               "stdout": out.strip()[-300:], "stderr": err.strip()[-500:]}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if p.returncode != 0: print(f"  {j['name']} stripe {k} FAILED rc={p.returncode}: {err.strip()[-300:]}")

    running = {}                       # (job name, stripe) -> (job, stripe, process)
    while True:
        remaining = a.wall - (time.time() - t0)
        for key, (j, k, p) in list(running.items()):
            if p.poll() is not None: record(j, k, p); del running[key]
        if remaining >= 25 and len(running) < N:
            for j, k in pending_pairs():
                if len(running) >= N: break
                if (j["name"], k) in running: continue
                running[(j["name"], k)] = (j, k, launch(j, k, max(10.0, min(a.budget, remaining - 12))))
        if not running: break
        time.sleep(1)
    all_done, lines = status(W, J, N)
    print("\n".join(lines)); print("ALL COMPLETE" if all_done else f"INCOMPLETE (invoke again); this call used {time.time() - t0:.0f}s")
    return 0 if all_done else 2


if __name__ == "__main__":
    sys.exit(main())
