"""Tests of the shared scan conventions and of the eight scan scripts on a synthetic release.

Run from the repository root:  python3 -m unittest -v tests.test_scanners
(or `make test`). No protected data is needed; every scanner runs end to end on the twelve
synthetic admissions in tests/synth.py, and the joint-cell identity assertion is exercised
both on consistent output and on a deliberately corrupted copy.
"""
import csv, glob, json, os, shutil, subprocess, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from alarmreplay.scan_common import parse_time, runs_from, in_any, limit_lookup, crossings, run_overshoots, run_stats  # noqa: E402
from tests import synth  # noqa: E402


def naive_seconds(s):
    """A calendar-naive reading of a timestamp (every month 31 days, every year 372), used as the
    counterexample the Gregorian parser is tested against."""
    return ((int(s[0:4]) * 372 + int(s[5:7]) * 31 + int(s[8:10])) * 24 + int(s[11:13])) * 3600 + int(s[14:16]) * 60 + int(s[17:19])


def T(day, hms, off="+00:00"):
    return parse_time(f"{day} {hms}{off}")


# ----------------------------------------------------------------------------- parse_time
class TimeParsing(unittest.TestCase):
    def test_within_month_differences_match_the_naive_reading(self):
        a = "2019-11-03 04:05:06+00:00"; b = "2019-11-29 23:59:58+00:00"
        self.assertEqual(parse_time(b) - parse_time(a), naive_seconds(b) - naive_seconds(a))

    def test_end_of_a_thirty_day_month_is_two_seconds_not_a_day(self):
        self.assertEqual(T("2019-12-01", "00:00:00") - T("2019-11-30", "23:59:58"), 2)
        self.assertEqual(naive_seconds("2019-12-01 00:00:00+00:00") - naive_seconds("2019-11-30 23:59:58+00:00"), 86402)

    def test_end_of_february_common_year(self):
        self.assertEqual(T("2019-03-01", "00:00:00") - T("2019-02-28", "23:59:58"), 2)

    def test_leap_day(self):
        self.assertEqual(T("2020-03-01", "00:00:00") - T("2020-02-29", "23:59:58"), 2)
        self.assertEqual(T("2020-02-29", "00:00:00") - T("2020-02-28", "00:00:00"), 86400)
        with self.assertRaises(ValueError):
            parse_time("2019-02-29 00:00:00+00:00")

    def test_end_of_year(self):
        self.assertEqual(T("2020-01-01", "00:00:00") - T("2019-12-31", "23:59:58"), 2)

    def test_explicit_offsets_are_normalised(self):
        ref = T("2019-11-19", "06:18:31")
        self.assertEqual(parse_time("2019-11-19T06:18:31Z"), ref)
        self.assertEqual(parse_time("2019-11-19 01:18:31-05:00"), ref)
        self.assertEqual(parse_time("2019-11-19 11:48:31+05:30"), ref)
        self.assertEqual(parse_time("2019-11-19 11:48:31+0530"), ref)

    def test_naive_timestamps_stay_on_the_recorded_coordinate(self):
        # no zone is inferred, and in particular nothing is done about daylight saving: a naive
        # stamp on a transition date is exactly where it was written
        self.assertEqual(parse_time("2019-11-03 06:18:31"), T("2019-11-03", "06:18:31"))
        self.assertEqual(parse_time("2019-03-10 02:30:00"), T("2019-03-10", "02:30:00"))

    def test_fractional_seconds_are_truncated(self):
        self.assertEqual(parse_time("2019-11-19 06:18:31.999+00:00"), T("2019-11-19", "06:18:31"))

    def test_malformed_timestamps_are_rejected(self):
        for bad in ("2019-11-19 06:18", "2019/11/19 06:18:31", "2019-11-19 06:18:31+99:00",
                    "2019-11-19 06:18:31.+00:00", "2019-13-01 00:00:00+00:00", ""):
            with self.assertRaises(ValueError, msg=bad):
                parse_time(bad)


# ----------------------------------------------------------------------------- runs, intervals, limits
class RunsAndLimits(unittest.TestCase):
    def test_runs_from_on_a_short_series(self):
        ts = [0, 2, 4, 10, 12, 30, 40, 42]
        self.assertEqual(runs_from(ts, 4), [(0, 4), (10, 12), (30, 30), (40, 42)])
        self.assertEqual(runs_from([], 4), [])
        self.assertEqual(runs_from([5], 4), [(5, 5)])

    def test_a_run_across_a_month_end_is_not_split(self):
        stamps = ["2019-11-30 23:59:56+00:00", "2019-11-30 23:59:58+00:00", "2019-12-01 00:00:00+00:00", "2019-12-01 00:00:02+00:00"]
        self.assertEqual(len(runs_from(sorted(parse_time(x) for x in stamps), 4)), 1)
        self.assertEqual(len(runs_from(sorted(naive_seconds(x) for x in stamps), 4)), 2)

    def test_in_any_on_closed_intervals(self):
        iv = [(10, 14), (20, 20)]; idx = [10, 20]
        for t, want in ((9, False), (10, True), (14, True), (15, False), (20, True), (21, False)):
            self.assertEqual(in_any(t, iv, idx), want, t)

    def test_limit_lookup_carries_forward_and_back(self):
        L = [(10, None, "120"), (20, None, "x"), (30, None, "110")]
        lim = limit_lookup(L, 2)
        self.assertEqual(lim(5), 120.0)     # before the first record: the first parsable value after it
        self.assertEqual(lim(10), 120.0)
        self.assertEqual(lim(25), 120.0)    # the unparsable record at 20 is skipped
        self.assertEqual(lim(30), 110.0)
        self.assertEqual(lim(99), 110.0)
        self.assertIsNone(limit_lookup([(10, None, None)], 2)(10))


# ----------------------------------------------------------------------------- overshoot
class Overshoot(unittest.TestCase):
    def setUp(self):
        self.sil = []; self.silidx = []

    def test_contemporaneous_limit_inside_an_excursion(self):
        L = [(2, None, "120"), (12, None, "110")]; lim = limit_lookup(L, 2)
        vv = [(t, 125.0) for t in range(0, 22, 2)]
        cross, exc = crossings(vv, self.sil, self.silidx, lim, "high")
        runs = runs_from(cross, 4)
        self.assertEqual(runs, [(0, 20)])
        self.assertEqual(run_overshoots(cross, exc, runs), [15.0])   # 125 - 110, not 125 - 120

    def test_silenced_sample_in_a_bridged_gap_does_not_set_the_overshoot(self):
        lim = limit_lookup([(2, "90", None)], 1)
        vv = [(0, 85.0), (2, 86.0), (4, 70.0), (6, 87.0), (8, 86.0)]
        sil = [(4, 4)]; silidx = [4]
        cross, exc = crossings(vv, sil, silidx, lim, "low")
        self.assertEqual(cross, [0, 2, 6, 8])
        runs = runs_from(cross, 4)
        self.assertEqual(runs, [(0, 8)])
        self.assertEqual(run_overshoots(cross, exc, runs), [5.0])

    def test_noncrossing_intermediate_sample_is_ignored(self):
        lim = limit_lookup([(2, "8", None)], 1)
        vv = [(0, 7.0), (2, 7.0), (4, 9.0), (6, 6.0), (8, 7.0)]
        cross, exc = crossings(vv, self.sil, self.silidx, lim, "low")
        self.assertEqual(cross, [0, 2, 6, 8])
        self.assertEqual(run_overshoots(cross, exc, runs_from(cross, 4)), [2.0])

    def test_overshoot_is_not_rounded(self):
        lim = limit_lookup([(2, "90", None)], 1)
        vv = [(0, 88.0), (2, 86.0004), (4, 88.0)]
        cross, exc = crossings(vv, self.sil, self.silidx, lim, "low")
        ov = run_overshoots(cross, exc, runs_from(cross, 4))[0]
        self.assertAlmostEqual(ov, 3.9996, places=9)
        self.assertNotEqual(round(ov, 3), ov)

    def test_shift_moves_the_limit(self):
        lim = limit_lookup([(2, "8", None)], 1)
        vv = [(0, 9.0), (2, 8.0)]
        self.assertEqual(crossings(vv, self.sil, self.silidx, lim, "low")[0], [2])
        self.assertEqual(crossings(vv, self.sil, self.silidx, lim, "low", shift=1.0)[0], [0, 2])   # looser low limit 9
        self.assertEqual(crossings(vv, self.sil, self.silidx, lim, "low", shift=-1.0)[0], [])     # tighter limit 7

    def test_run_stats_over_exactly_the_crossing_samples(self):
        lim = limit_lookup([(2, "90", None)], 1)
        vv = [(0, 85.0), (2, 86.0), (4, 70.0), (6, 87.0), (8, 86.0), (30, 80.0)]
        sil = [(4, 4)]; silidx = [4]
        cross, exc = crossings(vv, sil, silidx, lim, "low")
        runs = runs_from(cross, 4)
        self.assertEqual(runs, [(0, 8), (30, 30)])
        stats = run_stats(cross, exc, runs)
        self.assertEqual(stats[0][0], 5.0); self.assertAlmostEqual(stats[0][1], (5 + 4 + 3 + 4) / 4); self.assertEqual(stats[0][2], 4)
        self.assertEqual(stats[1], (10.0, 10.0, 1))


# ----------------------------------------------------------------------------- the scanners, end to end
class Scanners(unittest.TestCase):
    """Every scan script run on the synthetic release, one stripe, then its output inspected."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="scanners_")
        cls.tel, cls.cohort, cls.split, cls.design, cls.work = synth.build(cls.tmp)
        cls.env = dict(os.environ, ALARM_WORK=cls.work, ALARM_TELEMETRY=cls.tel, ALARM_COHORT=cls.cohort, PYTHONPATH=REPO)
        cls.log = {}
        cls.scan("scan_alarm_epidemiology.py", "0", "1", "60")
        cls.scan("scan_delay_model.py", "FIT", "0", "1", "60")
        cls.scan("scan_delay_model.py", "TEST", "0", "1", "60")
        cls.scan("scan_onset_lead.py", "0", "1", "60")
        cls.scan("scan_linkage.py", "0", "1", "60")
        cls.scan("scan_linkage_features.py", "FIT", "4", "0", "0", "1", "60")
        cls.scan("scan_linkage_features.py", "FIT", "4", "2", "0", "1", "60")
        cls.scan("scan_linkage_features.py", "TUNE", "2", "0", "0", "1", "60")
        cls.scan("scan_linkage_features.py", "FIT", "1", "0", "0", "1", "60")   # the whole fitting split, for f1, f3 and f5
        cls.scan("scan_counterfactual_limits.py", "0", "1", "60")
        cls.scan("scan_counterfactual_limits.py", "0", "1", "60", PD_FULL="1")
        cls.scan("scan_counterfactual_joint.py", "0", "1", "60")
        cls.scan("scan_counterfactual_joint.py", "0", "1", "60", PD3_PHASE="2")
        cls.scan("scan_counterfactual_joint.py", "0", "1", "60", PD3_SPLIT="TUNE", PD3_SUB="2")
        cls.scan("scan_measurement_conventions.py", "0", "1", "60")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def scan(cls, script, *args, **envextra):
        env = dict(cls.env, **envextra)
        p = subprocess.run([sys.executable, os.path.join(REPO, "scan", script), *args], env=env, capture_output=True, text=True, cwd=REPO)
        key = (script,) + args + tuple(sorted(envextra.items()))
        cls.log[key] = p
        if p.returncode != 0 or not p.stdout.strip().startswith("DONE"):
            raise RuntimeError(f"{script} {args} {envextra}: rc={p.returncode}\n{p.stdout}\n{p.stderr}")
        return p

    def rows(self, rel):
        out = []
        for fp in sorted(glob.glob(os.path.join(self.work, rel))):
            out += list(csv.DictReader(open(fp)))
        return out

    def lk2(self, tag):
        return self.rows(f"policy/lk2_rows_{tag}_*.csv")

    def test_every_scanner_reported_done(self):
        self.assertEqual(len(self.log), 15)
        for k, p in self.log.items():
            self.assertTrue(p.stdout.strip().startswith("DONE"), k)

    def test_epidemiology_rows_cover_every_admission(self):
        rows = self.rows("alarm_epidemiology/a1_rows_*.csv")
        self.assertEqual({(r["aMRN"], r["aCSN"]) for r in rows}, {(m, c) for m, c, _, _ in self._adms()})
        self.assertFalse(any(r["alarm_base"] in ("__READ_ERROR__", "__NO_ALARM_FILE__") for r in rows))
        f0 = [r for r in rows if r["aMRN"] == "f0"][0]
        self.assertEqual(int(f0["n_runs"]), 1); self.assertEqual(int(f0["run_seconds"]), 8)   # one annunciation run across the month end
        self.assertEqual(int(f0["adm_last_s"]) - int(f0["adm_first_s"]), 6)

    def _adms(self):
        return [(r["aMRN"], r["aCSN"], None, None) for r in csv.DictReader(open(self.cohort))]

    def test_month_end_excursion_is_one_excursion(self):
        r = [x for x in self.lk2("FIT4p0") if x["aMRN"] == "f0"]
        self.assertEqual(len(r), 1)
        r = r[0]
        self.assertEqual((int(r["D_s"]), r["A"], int(r["L_s"]), int(r["E_s"]), int(r["S_s"]), int(r["K"])), (12, "1", 2, -2, 8, 1))
        self.assertEqual(float(r["over"]), 1.0)

    def test_limit_change_inside_an_excursion_uses_the_contemporaneous_limit(self):
        # f1 is FIT index 1, outside the every-fourth subsample: read through the whole-split feature scan
        r = [x for x in self.lk2("FIT1p0") if x["aMRN"] == "f1"]
        self.assertEqual(len(r), 1); r = r[0]
        self.assertEqual(int(r["D_s"]), 22); self.assertEqual(float(r["over"]), 15.0); self.assertEqual(float(r["lim"]), 120.0)
        chg = self.rows("policy/lk2_chg_FIT1p0_*.csv")
        c = [x for x in chg if x["aMRN"] == "f1"]
        self.assertEqual(len(c), 1); self.assertEqual((c[0]["direction"], float(c[0]["magnitude"])), ("tighter", 10.0))
        self.assertEqual(int(r["nchg_before"]), 0); self.assertEqual(int(r["t_rel_chg"]), -12)

    def test_silenced_sample_in_a_bridged_gap(self):
        r = [x for x in self.lk2("FIT4p2") if x["aMRN"] == "f2"]
        self.assertEqual(len(r), 1); r = r[0]
        self.assertEqual(int(r["D_s"]), 10); self.assertEqual(float(r["over"]), 5.0)
        self.assertEqual(int(r["nmiss"]), 1)   # the silenced sample is not a crossing sample of the excursion

    def test_leap_day_boundary_and_noncrossing_sample(self):
        r = [x for x in self.lk2("FIT1p0") if x["aMRN"] == "f3"]
        self.assertEqual(len(r), 1); r = r[0]
        self.assertEqual((int(r["D_s"]), r["A"], int(r["S_s"])), (10, "1", 6)); self.assertEqual(float(r["over"]), 1.0)

    def test_overshoot_written_unrounded(self):
        r = [x for x in self.lk2("FIT4p0") if x["aMRN"] == "f4"]
        self.assertEqual(len(r), 1)
        self.assertEqual(float(r[0]["over"]), 90.0 - 86.0004)      # the full-precision difference, not 4.0
        self.assertNotEqual(float(r[0]["over"]), round(90.0 - 86.0004, 3))

    def test_offset_bearing_measurements_link_to_utc_alarms(self):
        r = [x for x in self.lk2("FIT1p0") if x["aMRN"] == "f5"]
        self.assertEqual(len(r), 1); r = r[0]
        self.assertEqual((int(r["D_s"]), r["A"], int(r["L_s"]), int(r["S_s"])), (8, "1", 2, 4))
        tot = [x for x in self.rows("policy/lk2_tot_FIT1p0_*.csv") if x["aMRN"] == "f5"][0]
        self.assertEqual(int(tot["unl4"]), 0)

    def test_subsample_allocation_is_unchanged(self):
        fit = [m for m, _, s, _ in synth_adms() if s == "FIT"]
        self.assertEqual({r["aMRN"] for r in self.lk2("FIT4p0")}, {fit[0], fit[4]})
        self.assertEqual({r["aMRN"] for r in self.lk2("FIT4p2")}, {fit[2]})
        tune = [m for m, _, s, _ in synth_adms() if s == "TUNE"]
        self.assertEqual({r["aMRN"] for r in self.lk2("TUNE2p0")}, {tune[0], tune[2]})
        self.assertEqual({r["aMRN"] for r in self.rows("policy/lk_rows_*.csv")}, {fit[0], fit[4]})
        self.assertEqual({r["aMRN"] for r in self.rows("policy/pd2_rows_*.csv")}, {fit[0], fit[4]})
        self.assertEqual({r["aMRN"] for r in self.rows("policy/pd_rows_*.csv")}, set(fit))
        self.assertEqual({r["aMRN"] for r in self.rows("policy/pd3_rows_[0-9].csv")}, {fit[0], fit[4]})
        self.assertEqual({r["aMRN"] for r in self.rows("policy/pd3_rows_p2_*.csv")}, {fit[2]})
        self.assertEqual({r["aMRN"] for r in self.rows("policy/pd3_rows_TUNE2p0_*.csv")}, {tune[0], tune[2]})

    def test_full_split_replay_is_zero_offset_only(self):
        rows = self.rows("policy/pd_rows_*.csv")
        self.assertTrue(rows); self.assertEqual({r["offset_step"] for r in rows}, {"0"})
        self.assertEqual({r["offset_step"] for r in self.rows("policy/pd2_rows_*.csv")}, {"-1", "0", "1", "2", "3"})

    def test_joint_replay_bins_the_unrounded_overshoot(self):
        r = [x for x in self.rows("policy/pd3_rows_[0-9].csv") if x["aMRN"] == "f4" and x["offset_step"] == "0"][0]
        cells = [tuple(int(v) for v in e.split(":")) for e in r["joint"].split(";")]
        self.assertEqual(cells, [(4, 2, 1, 8)])   # duration cell 4 (8 s), overshoot bin [2,4), one excursion, 8 seconds

    def test_delay_model_rows_on_both_splits(self):
        fit = self.rows("delay_model/a2_FIT_rows_*.csv"); test = self.rows("delay_model/a2_TEST_rows_*.csv")
        self.assertEqual({r["aMRN"] for r in fit}, {m for m, _, s, _ in synth_adms() if s == "FIT"})
        self.assertEqual({r["aMRN"] for r in test}, {m for m, _, s, _ in synth_adms() if s == "TEST"})
        f0 = [r for r in fit if r["aMRN"] == "f0" and r["tau"] == "0" and r["g"] == "8"][0]
        self.assertEqual((int(f0["rec_runs"]), int(f0["match"]), int(f0["recon_runs"]), int(f0["recon_secs"]), int(f0["rec_secs"])), (1, 1, 1, 12, 8))

    def test_onset_lead_histogram(self):
        h = json.load(open(os.path.join(self.work, "comparison", "c1_onset_hist_0_1.json")))
        self.assertEqual(h["admissions_done"], h["admissions_total"])
        self.assertEqual(sum(h["n_matched"].values()), 6)   # one matched annunciation per fitting admission

    def test_convention_scan_on_the_held_out_split(self):
        rows = self.rows("sensitivity/s2_rows_*.csv")
        self.assertEqual({r["aMRN"] for r in rows}, {m for m, _, s, _ in synth_adms() if s == "TEST"})
        self.assertEqual(len({r["config"] for r in rows}), 11)

    def test_read_logs_are_empty_on_a_clean_release(self):
        for fp in glob.glob(os.path.join(self.work, "cache", "*_readlog_*.csv")):
            self.assertEqual(len(list(csv.DictReader(open(fp)))), 0, fp)

    def test_missing_admission_is_logged(self):
        # a cohort row whose files do not exist is recorded, not silently skipped
        cohort2 = os.path.join(self.tmp, "cohort_missing.csv"); shutil.copy(self.cohort, cohort2)
        with open(cohort2, "a") as f: f.write("zz,99,none\n")
        work2 = os.path.join(self.tmp, "work2"); shutil.copytree(os.path.join(self.work, "design"), os.path.join(work2, "design"))
        with open(os.path.join(work2, "design", "SPLIT_ASSIGNMENT.csv"), "a") as f: f.write("zz,FIT\n")
        for sub in ("policy", "cache"): os.makedirs(os.path.join(work2, sub))
        env = dict(self.env, ALARM_WORK=work2, ALARM_COHORT=cohort2)
        p = subprocess.run([sys.executable, os.path.join(REPO, "scan", "scan_linkage_features.py"), "FIT", "1", "0", "0", "1", "60"], env=env, capture_output=True, text=True, cwd=REPO)
        self.assertTrue(p.stdout.startswith("DONE"), p.stdout + p.stderr)
        log = list(csv.DictReader(open(os.path.join(work2, "cache", "lk2_readlog_FIT1p0_0.csv"))))
        self.assertEqual([(r["aMRN"], r["reason"]) for r in log], [("zz", "missing_file")])


class ResumableOutputs(unittest.TestCase):
    """A run stopped after rows reached the disk but before the state recorded them must not
    duplicate those rows when it resumes."""

    def test_rows_beyond_the_saved_position_are_discarded_on_resume(self):
        tmp = tempfile.mkdtemp(prefix="resume_")
        try:
            tel, cohort, split, design, work = synth.build(tmp)
            env = dict(os.environ, ALARM_WORK=work, ALARM_TELEMETRY=tel, ALARM_COHORT=cohort, PYTHONPATH=REPO)
            cmd = [sys.executable, os.path.join(REPO, "scan", "scan_linkage_features.py"), "FIT", "1", "0", "0", "1"]
            # a reference run, uninterrupted
            ref = os.path.join(tmp, "ref"); shutil.copytree(work, ref)
            p = subprocess.run(cmd + ["60"], env=dict(env, ALARM_WORK=ref), capture_output=True, text=True, cwd=REPO); assert p.stdout.startswith("DONE"), p.stderr
            ref_rows = open(os.path.join(ref, "policy", "lk2_rows_FIT1p0_0.csv")).read()
            # a first pass with a budget too small to finish: it saves its position and stops
            p = subprocess.run(cmd + ["0.0001"], env=env, capture_output=True, text=True, cwd=REPO); assert p.stdout.startswith("PARTIAL"), p.stdout + p.stderr
            st = json.load(open(os.path.join(work, "cache", "lk2_state_FIT1p0_0.json")))
            self.assertIn("pos", st); self.assertLess(st["i"], 6)
            out = os.path.join(work, "policy", "lk2_rows_FIT1p0_0.csv")
            # rows that reached the disk after that save, as a stopped process would leave them
            with open(out, "a") as f: f.write("zz,99,ecg-numLimit-heartRate-high,10,0,,,,0,120,1,1,,,0,0,0,\n" * 3)
            p = subprocess.run(cmd + ["60"], env=env, capture_output=True, text=True, cwd=REPO); assert p.stdout.startswith("DONE"), p.stdout + p.stderr
            self.assertEqual(open(out).read(), ref_rows)
            self.assertNotIn("zz", open(out).read())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def synth_adms():
    return [("f0", "10", "FIT", "none"), ("f1", "11", "FIT", "none"), ("f2", "12", "FIT", "oncology"), ("f3", "13", "FIT", "none"),
            ("f4", "14", "FIT", "none"), ("f5", "15", "FIT", "none"), ("t0", "20", "TUNE", "none"), ("t1", "21", "TUNE", "none"),
            ("t2", "22", "TUNE", "none"), ("t3", "23", "TUNE", "none"), ("s0", "30", "TEST", "none"), ("s1", "31", "TEST", "none")]


# ----------------------------------------------------------------------------- the joint-cell identity
class JointIdentity(unittest.TestCase):
    """tools/assert_joint_consistency.py on the synthetic outputs, consistent and corrupted."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="jointid_")
        cls.tel, cls.cohort, cls.split, cls.design, cls.work = synth.build(cls.tmp)
        env = dict(os.environ, ALARM_WORK=cls.work, ALARM_TELEMETRY=cls.tel, ALARM_COHORT=cls.cohort, PYTHONPATH=REPO)
        for args, extra in ((("scan_linkage_features.py", "FIT", "4", "0", "0", "2", "60"), {}),
                            (("scan_linkage_features.py", "FIT", "4", "0", "1", "2", "60"), {}),
                            (("scan_counterfactual_joint.py", "0", "2", "60"), {}),
                            (("scan_counterfactual_joint.py", "1", "2", "60"), {}),
                            (("scan_linkage_features.py", "TUNE", "2", "0", "0", "1", "60"), {}),
                            (("scan_counterfactual_joint.py", "0", "1", "60"), {"PD3_SPLIT": "TUNE", "PD3_SUB": "2"})):
            p = subprocess.run([sys.executable, os.path.join(REPO, "scan", args[0]), *args[1:]], env=dict(env, **extra), capture_output=True, text=True, cwd=REPO)
            assert p.stdout.startswith("DONE"), (args, p.stdout, p.stderr)
        cls.env = env

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def assertion(self, work, *args):
        p = subprocess.run([sys.executable, os.path.join(REPO, "tools", "assert_joint_consistency.py"), "--work", work, "--cohort", self.cohort, *args],
                           env=self.env, capture_output=True, text=True, cwd=REPO)
        return p.returncode, p.stdout + p.stderr

    def copy_work(self, name):
        dst = os.path.join(self.tmp, name); shutil.copytree(self.work, dst); return dst

    def test_consistent_outputs_pass(self):
        rc, out = self.assertion(self.work, "--baseline-tag", "FIT4p0", "--stripes", "2")
        self.assertEqual(rc, 0, out); self.assertIn("IDENTITY HOLDS", out)
        rc, out = self.assertion(self.work, "--baseline-tag", "TUNE2p0", "--replay-pattern", "pd3_rows_TUNE2p0_*.csv", "--stripes", "1")
        self.assertEqual(rc, 0, out)

    def test_corrupted_feature_bin_fails(self):
        w = self.copy_work("corrupt")
        fp = sorted(glob.glob(os.path.join(w, "policy", "lk2_rows_FIT4p0_*.csv")))[0]
        rows = list(csv.reader(open(fp))); hdr = rows[0]; k = hdr.index("over")
        for r in rows[1:]:
            if r[k] != "": r[k] = str(float(r[k]) + 100.0); break     # one excursion moved to another overshoot bin
        csv.writer(open(fp, "w", newline="")).writerows(rows)
        rc, out = self.assertion(w, "--baseline-tag", "FIT4p0", "--stripes", "2")
        self.assertNotEqual(rc, 0); self.assertIn("IDENTITY FAILS", out)

    def test_corrupted_duration_fails(self):
        w = self.copy_work("corrupt_d")
        fp = sorted(glob.glob(os.path.join(w, "policy", "lk2_rows_FIT4p0_*.csv")))[0]
        rows = list(csv.reader(open(fp))); k = rows[0].index("D_s")
        rows[1][k] = str(int(rows[1][k]) + 2)
        csv.writer(open(fp, "w", newline="")).writerows(rows)
        rc, out = self.assertion(w, "--baseline-tag", "FIT4p0", "--stripes", "2")
        self.assertNotEqual(rc, 0); self.assertIn("IDENTITY FAILS", out)

    def test_empty_inputs_fail(self):
        w = self.copy_work("empty")
        for fp in glob.glob(os.path.join(w, "policy", "lk2_rows_FIT4p0_*.csv")):
            open(fp, "w").write("aMRN,aCSN,channel,D_s,A,L_s,E_s,S_s,K,lim,over,meanexc,slope10,pre30,nmiss,nchg_before,chg_adm,t_rel_chg\n")
        rc, out = self.assertion(w, "--baseline-tag", "FIT4p0", "--stripes", "2")
        self.assertNotEqual(rc, 0); self.assertIn("empty", out.lower())

    def test_duplicate_replay_rows_fail(self):
        w = self.copy_work("dup")
        fp = sorted(glob.glob(os.path.join(w, "policy", "pd3_rows_[0-9].csv")))[0]
        lines = open(fp).read().splitlines(); open(fp, "w").write("\n".join(lines + [lines[1]]) + "\n")
        rc, out = self.assertion(w, "--baseline-tag", "FIT4p0", "--stripes", "2")
        self.assertNotEqual(rc, 0); self.assertIn("duplicate", out.lower())

    def test_incomplete_stripe_fails(self):
        w = self.copy_work("partial")
        st = os.path.join(w, "cache", "pd3_state_1.json"); json.dump({"i": 0}, open(st, "w"))
        rc, out = self.assertion(w, "--baseline-tag", "FIT4p0", "--stripes", "2")
        self.assertNotEqual(rc, 0); self.assertIn("incomplete", out.lower())

    def test_missing_stripe_fails(self):
        w = self.copy_work("missing")
        os.remove(os.path.join(w, "policy", "pd3_rows_1.csv"))
        rc, out = self.assertion(w, "--baseline-tag", "FIT4p0", "--stripes", "2")
        self.assertNotEqual(rc, 0)

    def test_rounded_overshoot_fails(self):
        # a feature scan that rounds the overshoot before binning is what the identity is meant to catch:
        # the f4 row rounds 3.9996 to 4.0 and moves to another bin
        w = self.copy_work("rounded")
        fp = sorted(glob.glob(os.path.join(w, "policy", "lk2_rows_FIT4p0_*.csv")))
        for f in fp:
            rows = list(csv.reader(open(f))); k = rows[0].index("over")
            for r in rows[1:]:
                if r[k] != "": r[k] = str(round(float(r[k]), 3))
            csv.writer(open(f, "w", newline="")).writerows(rows)
        rc, out = self.assertion(w, "--baseline-tag", "FIT4p0", "--stripes", "2")
        self.assertNotEqual(rc, 0); self.assertIn("IDENTITY FAILS", out)


if __name__ == "__main__":
    unittest.main()
