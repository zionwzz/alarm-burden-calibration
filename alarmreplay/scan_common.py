"""Conventions shared by every scan script.

The eight scan scripts under scan/ share four small routines -- a timestamp parser, the run
former, an interval-membership test and the carried-forward limit lookup -- so that every
scanner sees the same excursion:

* **Elapsed time is Gregorian.** `parse_time` returns true elapsed seconds across month, year
  and leap-day ends. An explicit offset (`+00:00`, `Z`, `-05:00`) is normalised to UTC; a naive
  timestamp is kept on the coordinate it was recorded in, and nothing is inferred about daylight
  saving. Fractional seconds are truncated so that the 2-second recording grid is untouched.

* **Overshoot is the maximal directed excess beyond the contemporaneous limit, over exactly
  the eligible crossing samples that define the excursion.** A silenced or inactivated sample
  bridged by the 4-second run gap does not enter the maximum, a limit change inside the
  excursion is honoured sample by sample, and the value is binned unrounded. `crossings` and
  `run_overshoots` implement the convention for the feature scanner and the joint replay
  scanner alike, which is what makes the duration-by-overshoot cell identity between the two
  exact.

Everything else -- the 2-second duration `e - s + 2`, the plausibility ranges, the 4- and
30-second merge gaps, the silencing-run gap, the linking window, the split and subsample
allocation -- lives in the scanners themselves.
"""
import bisect
import datetime as _dt

__all__ = ["parse_time", "runs_from", "in_any", "limit_lookup", "crossings",
           "run_overshoots", "run_stats", "ReadLog", "Resumable"]

_EPOCH_ORDINAL = _dt.date(1970, 1, 1).toordinal()
_DAY_CACHE = {}      # 'YYYY-MM-DD' -> seconds at 00:00:00 of that calendar day
_OFFSET_CACHE = {}   # offset suffix -> seconds east of UTC


def _day_seconds(date):
    v = _DAY_CACHE.get(date)
    if v is None:
        if len(date) != 10 or date[4] != "-" or date[7] != "-":
            raise ValueError(f"malformed date {date!r}")
        d = _dt.date(int(date[0:4]), int(date[5:7]), int(date[8:10]))   # validates month/day, leap days included
        v = (d.toordinal() - _EPOCH_ORDINAL) * 86400
        _DAY_CACHE[date] = v
    return v


def _offset_seconds(suffix):
    v = _OFFSET_CACHE.get(suffix)
    if v is None:
        if suffix in ("", None):
            v = 0                                  # naive: stay on the recorded coordinate
        elif suffix == "Z":
            v = 0
        else:
            sign = suffix[0]
            if sign not in "+-":
                raise ValueError(f"malformed offset {suffix!r}")
            body = suffix[1:].replace(":", "")
            if len(body) not in (2, 4) or not body.isdigit():
                raise ValueError(f"malformed offset {suffix!r}")
            hh = int(body[0:2]); mm = int(body[2:4]) if len(body) == 4 else 0
            if hh > 14 or mm > 59:
                raise ValueError(f"offset out of range {suffix!r}")
            v = (hh * 3600 + mm * 60) * (1 if sign == "+" else -1)
        _OFFSET_CACHE[suffix] = v
    return v


def parse_time(s):
    """Elapsed seconds of a recorded timestamp on the Gregorian calendar.

    Accepts ``YYYY-MM-DD HH:MM:SS`` or ``YYYY-MM-DDTHH:MM:SS``, optionally followed by a
    fractional second and an offset (``Z``, ``+HH:MM``, ``+HHMM``). The value is seconds since
    1970-01-01T00:00:00 on the timestamp's own coordinate: an explicit offset is removed, so
    every offset-bearing timestamp lands on UTC; a naive timestamp is left where it was
    recorded. Fractional seconds are dropped. Anything else raises ``ValueError``.
    """
    s = s.strip()
    if len(s) < 19 or s[10] not in " T" or s[13] != ":" or s[16] != ":":
        raise ValueError(f"malformed timestamp {s!r}")
    secs = _day_seconds(s[0:10]) + int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19])
    rest = s[19:]
    if rest:
        if rest[0] == ".":                         # fractional second: truncate
            k = 1
            while k < len(rest) and rest[k].isdigit():
                k += 1
            if k == 1:
                raise ValueError(f"malformed timestamp {s!r}")
            rest = rest[k:]
        secs -= _offset_seconds(rest)
    return secs


def runs_from(ts, gap):
    """Maximal runs of a sorted list of sample times in which consecutive samples are no more
    than ``gap`` seconds apart; each run is ``(first, last)``."""
    out = []
    if not ts:
        return out
    s = prev = ts[0]
    for t in ts[1:]:
        if t - prev > gap:
            out.append((s, prev)); s = t
        prev = t
    out.append((s, prev))
    return out


def in_any(t, iv, idx):
    """Whether ``t`` lies inside one of the sorted closed intervals ``iv``; ``idx`` is their
    list of start times."""
    k = bisect.bisect_right(idx, t) - 1
    return k >= 0 and iv[k][1] >= t


def limit_lookup(L, idx):
    """The carried-forward limit as a function of time.

    ``L`` is the sorted list of ``(time, low, high)`` limit records of one alarm on one
    admission, ``idx`` 1 for the low limit or 2 for the high. The limit at ``t`` is the most
    recent parsable value at or before ``t``, or, before the first such record, the first
    parsable value after it; ``None`` if the alarm never carried a limit.
    """
    Lt = [x[0] for x in L]

    def lim_at(t):
        k = bisect.bisect_right(Lt, t) - 1
        for j in range(max(k, 0), -1, -1):
            x = L[j][idx]
            if x is not None:
                try:
                    return float(x)
                except ValueError:
                    pass
        for j in range(k + 1, len(L)):
            x = L[j][idx]
            if x is not None:
                try:
                    return float(x)
                except ValueError:
                    pass
        return None

    return lim_at


def crossings(vv, sil, silidx, lim_at, dirn, shift=0.0):
    """The eligible crossing samples of one channel, with their directed excess.

    ``vv`` is the sorted list of ``(time, value)`` plausible measurements; ``sil`` the
    silencing runs with ``silidx`` their starts; ``lim_at`` the carried-forward limit;
    ``dirn`` ``"high"`` or ``"low"``; ``shift`` an offset added to the limit (a candidate
    limit). A sample is eligible when it is not silenced and a limit is in force at its
    time, and it crosses when its value is at or beyond the contemporaneous limit. Returns
    ``(times, excesses)``, aligned, with the excess ``value - limit`` for a high limit and
    ``limit - value`` for a low one, so that a crossing sample has excess ``>= 0``.
    """
    cross = []; exc = []
    for t, v in vv:
        if in_any(t, sil, silidx):
            continue
        th = lim_at(t)
        if th is None:
            continue
        th = th + shift
        ex = (v - th) if dirn == "high" else (th - v)
        if ex >= 0:
            cross.append(t); exc.append(ex)
    return cross, exc


def run_overshoots(cross, exc, runs):
    """Maximal directed excess of each run, over exactly its own crossing samples.

    ``runs`` must be the runs ``runs_from(cross, gap)`` formed from these same crossing
    times, so that they partition ``cross`` in order. The value is not rounded.
    """
    out = []; j0 = 0
    for s, e in runs:
        j1 = bisect.bisect_right(cross, e, j0)
        out.append(max(exc[j0:j1])); j0 = j1
    return out


def run_stats(cross, exc, runs):
    """For each run: ``(overshoot, mean excess, number of distinct crossing sample times)``,
    all over exactly the run's crossing samples."""
    out = []; j0 = 0
    for s, e in runs:
        j1 = bisect.bisect_right(cross, e, j0)
        seg = exc[j0:j1]
        out.append((max(seg), sum(seg) / len(seg), len(set(cross[j0:j1])))); j0 = j1
    return out


class ReadLog:
    """Append-only record of admissions a scan could not read in full.

    A finished scan must be able to say how much of its intended cohort it covered, so each
    scanner writes one line per skipped admission -- ``aMRN, aCSN, reason`` -- to a file beside
    its state file, and the runner reports the totals against the intended lists.
    """

    def __init__(self, path):
        import os
        self._new = not os.path.exists(path)
        self._f = open(path, "a", newline="")
        if self._new:
            self._f.write("aMRN,aCSN,reason\n")

    def note(self, m, a, reason):
        self._f.write(f"{m},{a},{reason}\n"); self._f.flush()

    def close(self):
        self._f.close()


class Resumable:
    """Output files whose contents never run ahead of the state file.

    A scanner appends rows as it goes and records its position in a small state file every few
    admissions. Rows reach the disk whenever the file buffer fills, which can be *before* the
    position is recorded; a process stopped in that window leaves rows on disk for admissions the
    state file does not know about, and the next run would scan them again and append them a
    second time. The state file therefore also records the byte length of every output file at
    the moment of the save, and a resumed run truncates each output to that length before
    appending, so rows and state agree by construction.
    """

    def __init__(self, state_path, *outputs):
        import json, os
        self.state_path = state_path; self.outputs = list(outputs)
        self.state = json.load(open(state_path)) if os.path.exists(state_path) else {"i": 0}
        pos = self.state.get("pos", {})
        self.files = {}; self._new = {}
        for p in self.outputs:
            if p in pos and os.path.exists(p) and os.path.getsize(p) > pos[p]:
                with open(p, "r+b") as fh:
                    fh.truncate(pos[p])
            self._new[p] = (not os.path.exists(p)) or os.path.getsize(p) == 0
            self.files[p] = open(p, "a", newline="")

    def is_new(self, path):
        return self._new[path]

    def save(self, i, **extra):
        import json, os
        for fh in self.files.values():
            fh.flush(); os.fsync(fh.fileno())
        st = {"i": i, "pos": {p: fh.tell() for p, fh in self.files.items()}}
        st.update(extra)
        json.dump(st, open(self.state_path + ".tmp", "w")); os.replace(self.state_path + ".tmp", self.state_path)
        self.state = st

    def close(self):
        for fh in self.files.values():
            fh.close()
