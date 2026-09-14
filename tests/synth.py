"""A small synthetic telemetry release for exercising the scan scripts end to end.

Twelve admissions on the release's own file layout -- one directory per patient, one per
admission, ``ALARMp<mrn>e<csn>.csv.gz`` and ``MEASUREMENTp<mrn>e<csn>.csv.gz`` -- with a
cohort table and a split assignment. Each admission is built to make one convention
observable:

  f0  FIT   an excursion and its annunciation in progress at 2019-11-30 24:00:00, the end of a
            30-day month: one 12-second excursion on the calendar, two 6-second pieces under
            a calendar-naive 31-day-month reading
  f1  FIT   a limit change from 120 to 110 inside a heart-rate-high excursion: the overshoot
            against the contemporaneous limit is 15, against the start limit 5
  f2  FIT   a silenced sample of 70 inside a bridged 4-second gap of an SpO2 excursion whose
            crossing samples never fall below 85: overshoot 5, not 20
  f3  FIT   a non-crossing sample bridged by the gap, on the leap-day boundary 2020-02-29 into
            2020-03-01
  f4  FIT   an excess of 3.9996 that rounds to 4.000: bin [2,4) unrounded, [4,8) rounded
  f5  FIT   measurements stamped in -05:00 and alarms in +00:00 for the same instants: linked
            on the calendar, five hours apart under a parser that ignores the offset
  t0..t3 TUNE  plain excursions for the tuning subsample
  s0, s1 TEST  plain excursions for the held-out scans
"""
import csv, gzip, json, os

HEADER_A = "pollTime,alarmName,abnormalFlags,inactivationState,sil,setLow,setHigh,chanValue\n"
HEADER_M = "pollTime,mesname,msite,muom,mtext\n"

RR = ("ecgResp-numLimit-respRate-low#1unknown", "ecgResp-respRate#1", "L,PN,SP", "{breath}/min")
HRH = ("ecg-numLimit-heartRate-high#1unknown", "ecg-heartRate#1", "H,PN,SP", "{beat}/min")
HRL = ("ecg-numLimit-heartRate-low#1unknown", "ecg-heartRate#1", "L,PN,SP", "{beat}/min")
SP = ("spO2-numLimit-satO2-low#1unknown", "spO2-satO2#1", "L,PN,SP", "%")


def stamp(day, hms, off="+00:00"):
    return f"{day} {hms}{off}"


def hms(sec):
    return f"{sec // 3600:02d}:{(sec // 60) % 60:02d}:{sec % 60:02d}"


def arow(t, ch, lo, hi, val, sil=False, state="enabled"):
    return f"{t},{ch[0]},\"{ch[2]}\",{state},{'True' if sil else 'None'},{lo},{hi},{val}\n"


def mrow(t, ch, val):
    return f"{t},{ch[1]},None,{ch[3]},{val}\n"


def write_admission(root, mrn, csn, alarm_rows, meas_rows):
    d = os.path.join(root, mrn, csn); os.makedirs(d, exist_ok=True)
    with gzip.open(os.path.join(d, f"ALARMp{mrn}e{csn}.csv.gz"), "wt") as f:
        f.write(HEADER_A); f.writelines(alarm_rows)
    with gzip.open(os.path.join(d, f"MEASUREMENTp{mrn}e{csn}.csv.gz"), "wt") as f:
        f.write(HEADER_M); f.writelines(meas_rows)


def plain(day, start, ch, lo, hi, vals, ann_from=1, ann_to=None):
    """A single excursion: measurement values `vals` on the 2-second grid from `start` seconds
    into `day`, annunciated (un-silenced alarm rows carrying the limits) from sample index
    `ann_from` to `ann_to` (exclusive; default all but the last)."""
    if ann_to is None: ann_to = len(vals) - 1
    A = []; M = []
    for k, v in enumerate(vals):
        t = stamp(day, hms(start + 2 * k)); M.append(mrow(t, ch, v))
        if ann_from <= k < ann_to: A.append(arow(t, ch, lo, hi, v))
    return A, M


def build(root):
    """Write the release under `root`; return (telemetry_dir, cohort_csv, split_csv, design_json, work)."""
    tel = os.path.join(root, "telemetry-data"); os.makedirs(tel, exist_ok=True)
    adms = []   # (mrn, csn, split, tier)

    # f0: RR low, limit 8, excursion 23:59:54 .. 00:00:04 across the end of November
    A = []; M = []
    for k, (day, s) in enumerate([("2019-11-30", 86394), ("2019-11-30", 86396), ("2019-11-30", 86398),
                                  ("2019-12-01", 0), ("2019-12-01", 2), ("2019-12-01", 4)]):
        t = stamp(day, hms(s)); M.append(mrow(t, RR, 7))
        if 1 <= k <= 4: A.append(arow(t, RR, 8, 35, 7))
    # a quiet sample well inside the month on each side so the file spans the boundary either way
    M.insert(0, mrow(stamp("2019-11-30", hms(86000)), RR, 14)); M.append(mrow(stamp("2019-12-01", hms(400)), RR, 14))
    write_admission(tel, "f0", "10", A, M); adms.append(("f0", "10", "FIT", "none"))

    # f1: HR high, limit 120 then 110 from 12:00:12, values 125 from 12:00:00 to 12:00:20
    A = []; M = []
    for k in range(11):
        s = 43200 + 2 * k; t = stamp("2019-11-19", hms(s)); M.append(mrow(t, HRH, 125))
        if 1 <= k <= 10: A.append(arow(t, HRH, 50, 120 if s < 43212 else 110, 125))
    M.insert(0, mrow(stamp("2019-11-19", hms(43000)), HRH, 80)); M.append(mrow(stamp("2019-11-19", hms(43400)), HRH, 80))
    write_admission(tel, "f1", "11", A, M); adms.append(("f1", "11", "FIT", "none"))

    # f2: SpO2 low, limit 90; crossing samples 85,86,_,87,86 with a silenced 70 in the gap
    A = []; M = []
    vals = [85, 86, 70, 87, 86]
    for k, v in enumerate(vals):
        s = 36000 + 2 * k; t = stamp("2019-11-19", hms(s)); M.append(mrow(t, SP, v))
        if k == 2: A.append(arow(t, SP, 90, 100, v, sil=True))
        elif k >= 1: A.append(arow(t, SP, 90, 100, v))
    M.insert(0, mrow(stamp("2019-11-19", hms(35000)), SP, 97)); M.append(mrow(stamp("2019-11-19", hms(37000)), SP, 97))
    write_admission(tel, "f2", "12", A, M); adms.append(("f2", "12", "FIT", "oncology"))

    # f3: RR low, limit 8, a 9 (non-crossing) bridged by the gap, on the leap-day boundary
    A = []; M = []
    pts = [("2020-02-29", 86396, 7), ("2020-02-29", 86398, 7), ("2020-03-01", 0, 9), ("2020-03-01", 2, 7), ("2020-03-01", 4, 7)]
    for k, (day, s, v) in enumerate(pts):
        t = stamp(day, hms(s)); M.append(mrow(t, RR, v))
        if 1 <= k <= 3: A.append(arow(t, RR, 8, 35, v))
    write_admission(tel, "f3", "13", A, M); adms.append(("f3", "13", "FIT", "none"))

    # f4: SpO2 low, limit 90, one value 86.0004 (excess 3.9996) among 88s
    A, M = plain("2019-11-19", 7200, SP, 90, 100, [88, 86.0004, 88, 88])
    write_admission(tel, "f4", "14", A, M); adms.append(("f4", "14", "FIT", "none"))

    # f5: HR low, limit 50, values 45 at 06:18:30..06:18:36 UTC; measurements stamped in -05:00
    A = []; M = []
    for k in range(4):
        s = 22710 + 2 * k
        M.append(mrow(stamp("2019-11-19", hms(s - 5 * 3600), "-05:00"), HRL, 45))
        if 1 <= k <= 2: A.append(arow(stamp("2019-11-19", hms(s)), HRL, 50, 150, 45))
    write_admission(tel, "f5", "15", A, M); adms.append(("f5", "15", "FIT", "none"))

    # tuning and test admissions: plain excursions of different lengths
    for j, (mrn, csn, split, ch, lo, hi, vals) in enumerate([
            ("t0", "20", "TUNE", RR, 8, 35, [7, 6, 7, 7, 7, 7]),
            ("t1", "21", "TUNE", HRH, 50, 120, [122, 125, 130, 122]),
            ("t2", "22", "TUNE", SP, 90, 100, [88, 87, 88, 89, 88, 88, 88]),
            ("t3", "23", "TUNE", HRL, 50, 150, [45, 44, 45]),
            ("s0", "30", "TEST", RR, 8, 35, [7, 7, 7, 7, 7]),
            ("s1", "31", "TEST", HRH, 50, 120, [124, 126, 124, 124, 124, 124])]):
        A, M = plain("2019-11-20", 3600 + 600 * j, ch, lo, hi, vals)
        write_admission(tel, mrn, csn, A, M); adms.append((mrn, csn, split, "none"))

    cohort = os.path.join(root, "cohort.csv")
    with open(cohort, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["aMRN", "aCSN", "tier"]); w.writerows([(m, c, t) for m, c, _, t in adms])
    work = os.path.join(root, "work"); os.makedirs(os.path.join(work, "design"), exist_ok=True)
    split = os.path.join(work, "design", "SPLIT_ASSIGNMENT.csv")
    with open(split, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["aMRN", "split"]); w.writerows([(m, s) for m, _, s, _ in adms])
    design = os.path.join(work, "design", "DESIGN.json")
    with open(design, "w") as f:
        json.dump({"validation_criteria_TEST": {"episode_recall_min": 0.9, "burden_ratio_range": [0.8, 1.25],
                                                "onset_within_10s_min": 0.8}}, f, indent=1)
    for sub in ("alarm_epidemiology", "delay_model", "comparison", "sensitivity", "policy", "diagnostics", "cache"):
        os.makedirs(os.path.join(work, sub), exist_ok=True)
    return tel, cohort, split, design, work
