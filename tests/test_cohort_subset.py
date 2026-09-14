"""Tests of the optional cohort restriction (alarmreplay/cohort.py).

Run from the repository root:  python3 -m unittest -v tests.test_cohort_subset
No protected data is needed.
"""
import csv
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from alarmreplay.cohort import restriction, load_subset  # noqa: E402


def write(path, rows, header=("aCSN",)):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r if isinstance(r, (list, tuple)) else [r])
    return path


class CohortRestriction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_unrestricted_admits_everything(self):
        keep, record = restriction("")
        self.assertFalse(record["restricted"])
        for a in ("1", "2", "anything"):
            self.assertTrue(keep({"aCSN": a}))

    def test_keeps_exactly_the_named_admissions(self):
        p = write(os.path.join(self.tmp, "s.csv"), ["10", "20", "30"])
        keep, record = restriction(p, "a tier")
        self.assertTrue(record["restricted"])
        self.assertEqual(record["n_admissions_named"], 3)
        self.assertEqual(record["cohort"], "a tier")
        self.assertEqual([a for a in ("5", "10", "20", "25", "30")
                          if keep({"aCSN": a})], ["10", "20", "30"])

    def test_ignores_other_columns_and_blank_rows(self):
        p = os.path.join(self.tmp, "wide.csv")
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["aMRN", "aCSN", "tier"])
            w.writerows([["7", "10", "strict_onc"], ["8", "", "strict_onc"], ["9", "20", "non_onc"]])
        self.assertEqual(load_subset(p), {"10", "20"})

    def test_whitespace_is_not_a_different_admission(self):
        p = write(os.path.join(self.tmp, "ws.csv"), [" 10 ", "20"])
        keep, _ = restriction(p)
        self.assertTrue(keep({"aCSN": "10"}))

    def test_a_file_without_an_admission_column_is_refused(self):
        p = write(os.path.join(self.tmp, "bad.csv"), ["a", "b"], header=("patient",))
        with self.assertRaises(SystemExit):
            load_subset(p)

    def test_an_empty_subset_is_refused(self):
        p = write(os.path.join(self.tmp, "empty.csv"), [])
        with self.assertRaises(SystemExit):
            load_subset(p)

    def test_environment_drives_the_default(self):
        p = write(os.path.join(self.tmp, "env.csv"), ["42"])
        os.environ["ALARM_COHORT_SUBSET"] = p
        os.environ["ALARM_COHORT_LABEL"] = "from the environment"
        try:
            keep, record = restriction()
            self.assertTrue(keep({"aCSN": "42"}))
            self.assertFalse(keep({"aCSN": "43"}))
            self.assertEqual(record["cohort"], "from the environment")
        finally:
            del os.environ["ALARM_COHORT_SUBSET"], os.environ["ALARM_COHORT_LABEL"]


if __name__ == "__main__":
    unittest.main()
