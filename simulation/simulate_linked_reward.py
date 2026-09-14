#!/usr/bin/env python3
"""Simulation of linked-reward calibration using independent patient clusters.

Gamma patient effects are divided by their known population means. The same
patient count and effect apply at baseline and candidate limits; duration and
feature draws are conditionally independent between limits. Durations are rounded
up to a two-second grid. Known mean rewards define the scientific contrast;
cell standardisation uses the same reusable estimator as the clinical analysis.

Population truths use four million independent pseudo-patients per scenario.
Their Monte Carlo standard errors and saved replicate intervals allow explicit
assessment of truth precision. Primary feature maps use independent selection
samples. The in-sample map, baseline-reward oracle and fixed-offset estimator are
secondary comparators. Scenario F changes the candidate reward, so its baseline
oracle is not an oracle for that changed reward.

The script writes JSON summaries and replicate-level CSV files. The
main simulation has 500 replicates, with 200 bootstrap resamples in the first
400; patient-count checks use 400 replicates throughout. All seeds are explicit.
The supplement's simulation section states the complete data-generating laws; the simulation
keys its scenarios A, B, C, D, E, F, G, I, J, which the manuscript letters A to I in the order
of its design table (`alarmreplay/scenarios.py`).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alarmreplay.linked_reward import (  # noqa: E402
    CellMap, feature_support_map, reward_rates, replayed_burden, standardized_burden,
    burden_contrast, support_diagnostics,
)

SEED = 20260903
# Explicit per-scenario seeds, so that the run reproduces from one invocation to the next.
SCENARIO_SEED = {"A": 101, "B": 202, "C": 303, "D": 404, "E": 505, "F": 606, "G": 707,
                 "I": 808, "J": 909}
#: One truth seed per scenario, unrelated integers, so that the truth Monte Carlo errors are
#: independent across scenarios rather than sharing a draw order.
TRUTH_SEED = {"A": 811_001, "B": 822_013, "C": 833_021, "D": 844_037, "E": 855_047,
              "F": 866_059, "G": 877_069, "I": 888_079, "J": 899_091}
#: Pseudo-patients per scenario for the population truth: at 4,000,000 the truth Monte Carlo
#: standard error is 0.008-0.023 percentage points, small against every bias reported.
TRUTH_PATIENTS = 4_000_000
#: Patients per vectorized block of the truth. Sets the peak memory of the truth, not its value.
TRUTH_BLOCK = 100_000

N_FEATURES = 4                     # overshoot bins
CMAP = CellMap(width=2.0, n_exact=30, tail_edges=(60.0, 120.0), tail_midpoints=(80.0, 180.0))
NB, NT, NC = CMAP.n_exact, CMAP.n_tail, CMAP.n_cells
DGRID = CMAP.representative_duration

THETA = 8.0                        # persistence clock: below it the device cannot annunciate at all
#: Overshoot composition at the limit in force, and under the candidate limit where it shifts.
PZ_BASE = np.array([0.25, 0.30, 0.25, 0.20])
PZ_SHIFT = np.array([0.45, 0.28, 0.17, 0.10])
#: Scenarios in which the candidate limit leaves the overshoot composition alone.
NO_COMPOSITION_SHIFT = ("A", "D", "E", "G")
#: Scenarios with stochastic non-annunciation, i.e. observed zeros above the persistence clock.
ZERO_INFLATED = ("I", "J")
#: Miss probability multiplier by overshoot class in scenario J: an excursion that barely crosses the
#: limit is the one the device is most likely to miss. Scenario I uses the single value MISS_D.
MISS_BY_FEATURE = np.array([0.65, 0.45, 0.25, 0.12])
MISS_D = 0.40
#: Duration profile of the miss probability: 1.0 at d = 2 s, falling to MISS_FLOOR for long
#: excursions. It never reaches zero, so an excursion of ANY duration can go unannunciated.
MISS_FLOOR = 0.20
MISS_DECAY = 30.0


# ----------------------------------------------------------------------- the data-generating law
def draw_durations(rng, k, scale, frailty, scenario, u):
    """Excursion durations, rounded UP to the 2-second recording grid.

    `2*ceil(g/2)` puts every duration on the same grid the exact cells of CMAP sit on, so a duration
    below 60 s is represented by its cell EXACTLY and the cell map loses nothing there.

    Scenario E changes TWO things relative to every other scenario, and the second is easy to miss.
    Here the limit in force is CLIPPED at 100 s: `np.clip` does not resample the excursions that
    would have run longer, it maps all of them to exactly 100 s, so the baseline duration law has a
    POINT MASS at 100 s carrying the whole upper tail. That is not a truncated draw and calling it
    "truncated", as this file did, misdescribed the law; the point mass sits in the pooled [60, 120)
    cell. In `draw_sample` and `true_percent_change` the candidate limit's duration scale is also
    raised from 11.0 to 20.0 (every other scenario loosens to 11.0, which SHORTENS excursions). The
    clip is what creates support failure in the identification sense -- the candidate puts replayed
    seconds in duration cells that hold no baseline excursion at all, where no reward is identified
    and the estimator can only report how much target mass it cannot reach. The raised scale is what
    makes the candidate lengthen rather than shorten excursions, and is why E is the only scenario
    whose truth is negative. Neither change is incidental and both belong in the scenario's
    description.
    """
    d = 2.0 * np.ceil(rng.gamma(1.6, scale * frailty ** 0.3, k) / 2.0)
    hi = 100.0 if (scenario == "E" and u == 0) else 400.0
    return np.clip(d, 2.0, hi)


def duration_scale(scenario, u):
    """Gamma scale of the duration law under limit `u`. E's candidate LENGTHENS, every other shortens."""
    if u == 0:
        return 14.0
    return 20.0 if scenario == "E" else 11.0


def feature_probabilities(scenario, u):
    """Overshoot-class probabilities under limit `u`."""
    if u == 0 or scenario in NO_COMPOSITION_SHIFT:
        return PZ_BASE
    return PZ_SHIFT


def _miss_probability(d, z, scenario):
    """P(the device fails to annunciate an excursion of duration d and overshoot class z).

    Zero in scenarios A-G, where the only zeros are the deterministic ones below the persistence
    clock. In I and J it is strictly positive at EVERY duration -- it falls from 1.0 x the class
    multiplier at d = 2 s towards MISS_FLOOR x it for long excursions, and never reaches zero -- so
    every duration cell above the clock holds observed zeros beside positive rewards. In I it does
    not depend on z, so duration transport still holds; in J it does, and the reward's whole
    dependence on overshoot runs through it.
    """
    d = np.asarray(d, dtype=float)
    if scenario not in ZERO_INFLATED:
        return np.zeros(d.shape)
    decay = MISS_FLOOR + (1.0 - MISS_FLOOR) * np.exp(-(d - 2.0) / MISS_DECAY)
    if scenario == "I":
        return MISS_D * decay
    return MISS_BY_FEATURE[np.asarray(z)] * decay


def _reward_components(d, z, scenario, limit):
    """(mean linked seconds GIVEN annunciation, P(no annunciation)) for an excursion.

    `draw_sample` needs the two separately -- it draws the annunciation indicator and then the
    positive reward -- while the estimand needs only their product. Keeping one function for both is
    what stops the generator and the estimand from drifting apart.
    """
    d = np.asarray(d, dtype=float)
    z = np.asarray(z)
    base = np.maximum(d - THETA, 0.0)
    if scenario in ("A", "E", "G", "F", "I", "J"):
        gain = np.full(d.shape, 0.85)
    elif scenario == "D":
        # within the pooled cells the ratio keeps rising, so a cell's mean is not its composition
        gain = np.clip(0.55 + 0.004 * d, 0.55, 0.95)
    else:
        gain = np.array([0.45, 0.70, 0.85, 0.92])[z]
    if scenario == "F" and limit != 0:
        gain = gain * 0.80          # the reward itself moves with the limit: transport fails
    return base * gain, _miss_probability(d, z, scenario)


def true_reward(d, z, scenario, limit):
    """a_0(d, z): expected linked annunciation seconds for an excursion of duration d, feature z.

    The device cannot annunciate at all before the persistence clock theta = 8 s, so an excursion at
    or below it has a reward of exactly zero; beyond that the reward grows with duration, and in the
    feature-dependent scenarios an excursion that barely crosses the limit (z = 0) earns less. In the
    zero-inflated scenarios the expectation also carries the `(1 - p_miss(d, z))` factor, because an
    excursion the device misses contributes an observed zero, not a missing value. That factor is
    HERE, in the estimand, and `draw_sample` reads it from the same function, so the two cannot
    disagree. This is the shape the application estimates, not a shape assumed by the estimator.
    """
    ann, pmiss = _reward_components(d, z, scenario, limit)
    return (1.0 - pmiss) * ann


def draw_sample(rng, n_patients, scenario):
    """One simulated sample: patient-level cell sums at the limit in force and under the candidate."""
    strong = scenario == "G"
    frail = rng.gamma(1.2 if strong else 4.0, 1.0, n_patients)
    frail /= (1.2 if scenario == "G" else 4.0)
    lam = 25.0 * frail
    n_exc = rng.poisson(lam)
    n_exc = np.maximum(n_exc, 1)

    S1 = np.zeros((n_patients, NC))                       # linked seconds, limit in force
    R1 = np.zeros((n_patients, NC))                       # carrier, limit in force
    S2 = np.zeros((n_patients, NC, N_FEATURES))
    R2 = np.zeros((n_patients, NC, N_FEATURES))
    N1 = np.zeros((n_patients, NC))                       # counts, for the support map
    N2 = np.zeros((n_patients, NC, N_FEATURES))
    Z1 = np.zeros(NC)                                     # observed-zero counts, for the diagnostics
    Z2 = np.zeros((NC, N_FEATURES))
    Cnt = {0: np.zeros((n_patients, NC)), 1: np.zeros((n_patients, NC))}
    Sec = {0: np.zeros((n_patients, NT)), 1: np.zeros((n_patients, NT))}
    Cnt2 = {0: np.zeros((n_patients, NC, N_FEATURES)), 1: np.zeros((n_patients, NC, N_FEATURES))}
    Sec2 = {0: np.zeros((n_patients, NC, N_FEATURES)), 1: np.zeros((n_patients, NC, N_FEATURES))}
    Aor = {0: 0.0, 1: 0.0}                                # exact int a_0 d nu_u on the realized sample
    n_zero = 0.0
    max_d_zero = 0.0

    for u in (0, 1):
        # loosening the limit shortens excursions and moves them toward the just-crossing feature bin
        scale = duration_scale(scenario, u)
        pz = feature_probabilities(scenario, u)
        for i in range(n_patients):
            k = n_exc[i]
            d = draw_durations(rng, k, scale, frail[i], scenario, u)
            z = rng.choice(N_FEATURES, size=k, p=pz)
            c = CMAP.index(d)
            if u == 0:
                ann, pmiss = _reward_components(d, z, scenario, u)
                # Gamma with mean `ann`: shape ann/3, scale 3, so E(Y | annunciated, d, z) = ann up
                # to the 1e-9 guard, and E(Y | d, z) = (1 - p_miss) ann = a_0(d, z).
                y = rng.gamma(np.maximum(ann, 1e-9) / 3.0 + 1e-9, 3.0)
                y = np.where(ann <= 0, 0.0, y)
                if scenario in ZERO_INFLATED:
                    # stochastic non-annunciation: an observed zero at ANY duration, which is what
                    # puts zeros and positive rewards in the same cell.
                    y = np.where(rng.random(k) < pmiss, 0.0, y)
                zero = y <= 0.0
                if zero.any():
                    n_zero += float(zero.sum())
                    max_d_zero = max(max_d_zero, float(d[zero].max()))
                    np.add.at(Z1, c[zero], 1.0)
                    np.add.at(Z2, (c[zero], z[zero]), 1.0)
                np.add.at(S1[i], c, y)
                np.add.at(N1[i], c, 1.0)
                np.add.at(S2[i], (c, z), y)
                np.add.at(N2[i], (c, z), 1.0)
                np.add.at(R1[i], c, np.where(c < NB, 1.0, d))
                np.add.at(R2[i], (c, z), np.where(c < NB, 1.0, d))
            Aor[u] += float(true_reward(d, z, scenario, 0).sum())
            np.add.at(Cnt[u][i], c, 1.0)
            np.add.at(Cnt2[u][i], (c, z), 1.0)
            tail = c >= NB
            if tail.any():
                np.add.at(Sec[u][i], c[tail] - NB, d[tail])
                np.add.at(Sec2[u][i], (c[tail], z[tail]), d[tail])
    n_base = float(N1.sum())
    n1 = N1.sum(0)
    n2 = N2.sum(0)
    zero_diag = dict(
        n_excursions=n_base,
        zero_share=(n_zero / n_base) if n_base else float("nan"),
        max_duration_with_zero=max_d_zero,
        mixing_duration_cells=float(((Z1 > 0) & (n1 - Z1 > 0)).sum()),
        mixing_joint_cells=float(((Z2 > 0) & (n2 - Z2 > 0)).sum()),
        share_excursions_in_mixing_duration_cells=(
            float(n1[(Z1 > 0) & (n1 - Z1 > 0)].sum() / n_base) if n_base else float("nan")),
    )
    return dict(S1=S1, R1=R1, N1=N1, S2=S2, R2=R2, N2=N2, Cnt=Cnt, Sec=Sec, Cnt2=Cnt2, Sec2=Sec2,
                n_exc=n_exc, oracle=Aor, zero_diag=zero_diag)


def carriers(Cnt_u, Sec_u):
    """Target carrier by cell: counts in the exact cells, replayed seconds in the pooled cells."""
    return np.concatenate([Cnt_u[:NB], Sec_u])


def carriers2(Cnt2_u, Sec2_u):
    R = Cnt2_u.copy()
    R[NB:] = Sec2_u[NB:]
    return R


# ---------------------------------------------------------------------------------- the truth
_TRUTH_CACHE: dict = {}


def true_percent_change(scenario, n_patients=None, seed=None, block=None):
    """Monte Carlo population contrast from independent patient clusters.

    The four-million-patient calculation uses the same duration, feature, count,
    and mean-reward law as draw_sample. Independent gamma effects are divided by
    their population mean, not a realised sample mean. Per-patient reward sums
    yield a ratio and its delta-method Monte Carlo standard error. Block size is
    a computational setting and is recorded with the calculation.
    """
    # Resolved here rather than in the signature so that TRUTH_PATIENTS stays a live module-level
    # setting: a default argument would fix its value at import time.
    if n_patients is None:
        n_patients = TRUTH_PATIENTS
    if block is None:
        block = TRUTH_BLOCK
    if seed is None:
        seed = TRUTH_SEED[scenario]
    key = (scenario, int(n_patients), int(seed))
    if key in _TRUTH_CACHE:
        return _TRUTH_CACHE[key]
    rng = np.random.default_rng(seed)
    frail = rng.gamma(1.2 if scenario == "G" else 4.0, 1.0, n_patients)
    frail /= (1.2 if scenario == "G" else 4.0)
    n_exc = np.maximum(rng.poisson(25.0 * frail), 1)
    per = {0: np.empty(n_patients), 1: np.empty(n_patients)}
    for u in (0, 1):
        scale = duration_scale(scenario, u)
        cum = np.cumsum(feature_probabilities(scenario, u))[:-1]
        hi = 100.0 if (scenario == "E" and u == 0) else 400.0
        for s in range(0, n_patients, block):
            e = min(s + block, n_patients)
            k = n_exc[s:e]
            tot = int(k.sum())
            pid = np.repeat(np.arange(e - s), k)
            d = 2.0 * np.ceil(rng.gamma(1.6, scale * frail[s:e][pid] ** 0.3) / 2.0)
            np.clip(d, 2.0, hi, out=d)
            z = np.searchsorted(cum, rng.random(tot))
            per[u][s:e] = np.bincount(pid, weights=true_reward(d, z, scenario, u),
                                      minlength=e - s)
    a0, a1 = per[0], per[1]
    B0, B1 = float(a0.sum()), float(a1.sum())
    m = a0.size
    r = B1 / B0
    g = (a1 - r * a0) / (a0.mean() * m ** 0.5)
    out = (100.0 * (1.0 - r), 100.0 * float(np.std(g, ddof=1)))
    _TRUTH_CACHE[key] = out
    return out


def true_percent_change_looped(seed, scenario, n_patients=20000):
    """A patient-by-patient truth, kept only so the vectorized routine can be checked against it.

    It reads the duration scale and the feature probabilities from the shared helpers. It is not
    used by any run; `--verify-truth-vectorization` calls it to show that the vectorized routine
    draws the same law.
    """
    rng = np.random.default_rng(seed)
    B = {0: 0.0, 1: 0.0}
    per_patient = {0: [], 1: []}
    frail = rng.gamma(1.2 if scenario == "G" else 4.0, 1.0, n_patients)
    frail /= (1.2 if scenario == "G" else 4.0)
    n_exc = np.maximum(rng.poisson(25.0 * frail), 1)
    for u in (0, 1):
        scale = duration_scale(scenario, u)
        pz = feature_probabilities(scenario, u)
        tot = 0.0
        for i in range(n_patients):
            k = n_exc[i]
            d = draw_durations(rng, k, scale, frail[i], scenario, u)
            z = rng.choice(N_FEATURES, size=k, p=pz)
            v = float(true_reward(d, z, scenario, u).sum())
            per_patient[u].append(v)
            tot += v
        B[u] = tot
    a0 = np.asarray(per_patient[0]); a1 = np.asarray(per_patient[1])
    m = a0.size
    r = B[1] / B[0]
    g = (a1 - r * a0) / (a0.mean() * m ** 0.5)
    return 100.0 * (1.0 - r), 100.0 * float(np.std(g, ddof=1))


def verify_truth_vectorization(scenarios, n_seeds=30, n_patients=20000, base_seed=20261894,
                               stride=7919, out=sys.stdout):
    """Show that `true_percent_change` draws the same law as `true_percent_change_looped`.

    The two routines consume randomness in a different order, so they do not agree replicate by
    replicate at a shared seed and no equality test is meaningful. What is meaningful is whether they
    have the same sampling distribution, so this runs both at `n_seeds` seeds and reports

      * the difference in means against its two-sample standard error (a z that should be O(1));
      * each version's empirical standard deviation across seeds, which should agree with each other
        AND with the delta-method standard error each version reports, since that standard error is
        an estimate of exactly this spread;
      * the ratio of the two variances.

    Returns a list of dicts, one per scenario, so a caller can assert on the numbers.
    """
    rows = []
    for sc in scenarios:
        L = np.array([true_percent_change_looped(base_seed + stride * j, sc, n_patients)
                      for j in range(n_seeds)])
        V = np.array([true_percent_change(sc, n_patients, seed=base_seed + stride * j)
                      for j in range(n_seeds)])
        lv, ls, vv, vs = L[:, 0], L[:, 1], V[:, 0], V[:, 1]
        dm = float(vv.mean() - lv.mean())
        se = float(np.sqrt(lv.var(ddof=1) / n_seeds + vv.var(ddof=1) / n_seeds))
        hp = true_percent_change(sc)
        row = dict(scenario=sc, n_seeds=n_seeds, n_patients=n_patients,
                   mean_looped=float(lv.mean()), mean_vector=float(vv.mean()),
                   diff=dm, se_diff=se, z=dm / se if se else float("nan"),
                   sd_looped=float(lv.std(ddof=1)), sd_vector=float(vv.std(ddof=1)),
                   mean_delta_se_looped=float(ls.mean()), mean_delta_se_vector=float(vs.mean()),
                   var_ratio=float(vv.var(ddof=1) / lv.var(ddof=1)),
                   high_precision=hp[0], high_precision_mcse=hp[1])
        rows.append(row)
        print(f"{sc}: means  looped {row['mean_looped']:9.5f}  vector {row['mean_vector']:9.5f}  "
              f"diff {dm:+.5f} +- {se:.5f}  z={row['z']:+.2f}", file=out)
        print(f"   spread across seeds  looped SD {row['sd_looped']:.5f}  vector SD "
              f"{row['sd_vector']:.5f}  var ratio {row['var_ratio']:.3f}", file=out)
        print(f"   mean delta-method SE  looped {row['mean_delta_se_looped']:.5f}  vector "
              f"{row['mean_delta_se_vector']:.5f}   (these estimate the SDs above)", file=out)
        print(f"   high-precision truth ({TRUTH_PATIENTS:,} patients) "
              f"{row['high_precision']:.6f} +- {row['high_precision_mcse']:.6f}", file=out)
    return rows


# ------------------------------------------------------------------------------- the estimators
def fit_all(sample, feature_map, nmin, pre_map=None, diagnostics=True):
    """Point estimates of the percentage change under each estimator, on one sample.

    `feature_map` is the duration-by-overshoot support map used by LRC-dz. In the in-sample arm it
    was selected on this very sample; in the off-sample arm it was selected on an independent draw.
    Either way it is a fixed array here, so the same function serves both arms and every bootstrap
    replicate. Passing `pre_map` additionally evaluates LRC-dz with that second, off-sample map on
    the same data, which is how the two arms are compared on identical samples.

    `diagnostics=False` skips the support diagnostics, which no bootstrap replicate uses. It changes
    nothing that is reported and consumes no randomness; it is there only so the bootstrap does not
    pay for a diagnostic it discards.
    """
    S1 = sample["S1"].sum(0); R1 = sample["R1"].sum(0); N1 = sample["N1"].sum(0)
    S2 = sample["S2"].sum(0); R2 = sample["R2"].sum(0)
    cnt0 = sample["Cnt"][0].sum(0); sec0 = sample["Sec"][0].sum(0)
    cnt1 = sample["Cnt"][1].sum(0); sec1 = sample["Sec"][1].sum(0)
    out = {}

    rep0 = replayed_burden(CMAP, cnt0[:NB], sec0)
    rep1 = replayed_burden(CMAP, cnt1[:NB], sec1)
    out["replay"] = burden_contrast(rep1, rep0)

    beta = reward_rates(S1, R1)
    b0 = standardized_burden(beta, carriers(cnt0, sec0))
    b1 = standardized_burden(beta, carriers(cnt1, sec1))
    out["lrc_d"] = burden_contrast(b1, b0)

    t0 = carriers2(sample["Cnt2"][0].sum(0), sample["Sec2"][0].sum(0))
    t1 = carriers2(sample["Cnt2"][1].sum(0), sample["Sec2"][1].sum(0))
    ok = feature_map & (R2 > 0)
    beta2 = reward_rates(S2, np.where(ok, R2, 0.0), fallback=beta)
    out["lrc_dz"] = burden_contrast(standardized_burden(beta2, t1), standardized_burden(beta2, t0))

    if pre_map is not None:
        okp = pre_map & (R2 > 0)
        beta2p = reward_rates(S2, np.where(okp, R2, 0.0), fallback=beta)
        out["lrc_dz_pre"] = burden_contrast(standardized_burden(beta2p, t1),
                                            standardized_burden(beta2p, t0))

    if diagnostics:
        out["_support"] = support_diagnostics(CMAP, N1, cnt1, sec1, min_for_precision=20)
        out["_fallback_share"] = float((~feature_map).mean())
        out["_empty_map_cells"] = int((feature_map & (R2 <= 0)).sum())
        if pre_map is not None:
            out["_fallback_share_pre"] = float((~pre_map).mean())
            out["_map_disagreement"] = float((pre_map != feature_map).mean())
    return out


def oracle_percent_change(sample, scenario):
    """Integrate the known baseline reward over each realised excursion measure.

    It has no cell approximation but remains misspecified for the changed
    candidate reward in scenario F.
    """
    return burden_contrast(sample["oracle"][1], sample["oracle"][0])


def offset_percent_change(sample):
    """Fixed-offset persistence comparator: solve for the offset that reproduces the linked seconds."""
    S1 = sample["S1"].sum(0); N1 = sample["N1"].sum(0)
    cnt0 = sample["Cnt"][0].sum(0); sec0 = sample["Sec"][0].sum(0)
    cnt1 = sample["Cnt"][1].sum(0); sec1 = sample["Sec"][1].sum(0)
    tot_secs0 = replayed_burden(CMAP, cnt0[:NB], sec0)
    target = S1.sum() / max(tot_secs0, 1e-9)

    def r(th):
        exact = (cnt0[:NB] * np.maximum(DGRID[:NB] - th, 0.0)).sum()
        dbar = np.divide(sec0, np.maximum(cnt0[NB:], 1e-9))
        tail = (cnt0[NB:] * np.maximum(dbar - th, 0.0)).sum()
        return (exact + tail) / max(tot_secs0, 1e-9)

    lo, hi = 0.0, 120.0
    if r(lo) < target:
        return float("nan")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if r(mid) > target:
            lo = mid
        else:
            hi = mid
    th = 0.5 * (lo + hi)

    def burden(cnt, sec):
        exact = (cnt[:NB] * np.maximum(DGRID[:NB] - th, 0.0)).sum()
        dbar = np.divide(sec, np.maximum(cnt[NB:], 1e-9))
        return float(exact + (cnt[NB:] * np.maximum(dbar - th, 0.0)).sum())

    return burden_contrast(burden(cnt1, sec1), burden(cnt0, sec0))


# -------------------------------------------------------------------------------------- the run
def run_scenario(scenario, n_reps, n_patients, n_boot, seed, dgp=None, pre_arm=True, record=None):
    """One scenario: point estimation over `n_reps` samples, coverage over the first 400 of them.

    `dgp` names the data-generating law when it differs from the scenario label (the sample-size
    sweep runs scenario A's law under the label H). `pre_arm` adds the prespecified-map arm, which
    costs one extra independent sample per replicate. `record`, when given, is a list that collects
    one row per replicate and estimator -- the point estimate and, where there is one, the interval
    endpoints -- so that the run can be rescored against a refined truth without being repeated.
    """
    dgp = dgp or scenario
    truth, truth_se = true_percent_change(dgp)
    rng_master = np.random.default_rng(seed + SCENARIO_SEED[scenario])
    # A SECOND stream, independent of rng_master, supplies the prespecified arm's selection samples.
    # Keeping it separate is deliberate: the analysis samples, the bootstrap draws and therefore every
    # number the other estimators report are exactly what they would be with the arm switched off, so
    # adding this arm cannot move the rest of the table.
    rng_select = np.random.default_rng(seed + 4177 + SCENARIO_SEED[scenario])
    keys = ["replay", "lrc_d", "lrc_dz", "baseline_oracle", "offset"]
    if pre_arm:
        keys.insert(3, "lrc_dz_pre")
    est = {k: [] for k in keys}
    se_d, se_dz, se_pre = [], [], []
    ci_dz, ci_d, ci_pre = [], [], []
    unsup, fallback, empty, fb_pre, disagree = [], [], [], [], []
    zdiag = []
    nmin = 5
    n_cov = min(n_reps, 400)

    for rep in range(n_reps):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        sample = draw_sample(rng, n_patients, dgp)
        fmap = feature_support_map(sample["N2"].sum(0), nmin)
        pmap = None
        if pre_arm:
            # the prespecified map: selected on an INDEPENDENT draw of the same size from the same
            # law, then applied to this sample unchanged and held fixed through the bootstrap. This
            # is the design the application uses when a tuning split is available, and it is the only
            # arm whose map is not a function of the data it is applied to.
            sel = draw_sample(np.random.default_rng(rng_select.integers(0, 2**32 - 1)),
                              n_patients, dgp)
            pmap = feature_support_map(sel["N2"].sum(0), nmin)
            del sel
        point = fit_all(sample, fmap, nmin, pre_map=pmap)
        est["replay"].append(point["replay"])
        est["lrc_d"].append(point["lrc_d"])
        est["lrc_dz"].append(point["lrc_dz"])
        if pre_arm:
            est["lrc_dz_pre"].append(point["lrc_dz_pre"])
            fb_pre.append(point["_fallback_share_pre"])
            disagree.append(point["_map_disagreement"])
        est["baseline_oracle"].append(oracle_percent_change(sample, dgp))
        est["offset"].append(offset_percent_change(sample))
        unsup.append(point["_support"]["share_unsupported"])
        fallback.append(point["_fallback_share"])
        empty.append(point["_empty_map_cells"])
        zdiag.append(sample["zero_diag"])

        rep_ci = {}
        if n_boot and rep < n_cov:
            bd, bdz, bpre = [], [], []
            bidx = np.random.default_rng(rng.integers(0, 2**32 - 1))
            for _ in range(n_boot):
                idx = bidx.integers(0, n_patients, n_patients)
                sub = {}
                for k, v in sample.items():
                    if isinstance(v, np.ndarray):
                        sub[k] = v[idx]
                    elif isinstance(v, dict):
                        sub[k] = {u: (w[idx] if isinstance(w, np.ndarray) else w)
                                  for u, w in v.items()}
                    else:
                        sub[k] = v
                q = fit_all(sub, fmap, nmin, pre_map=pmap, diagnostics=False)
                bd.append(q["lrc_d"]); bdz.append(q["lrc_dz"])
                if pre_arm:
                    bpre.append(q["lrc_dz_pre"])
            bd = np.asarray(bd, float); bdz = np.asarray(bdz, float)
            se_d.append(float(np.nanstd(bd, ddof=1)))
            se_dz.append(float(np.nanstd(bdz, ddof=1)))
            ci_dz.append(tuple(np.nanpercentile(bdz, [2.5, 97.5])))
            ci_d.append(tuple(np.nanpercentile(bd, [2.5, 97.5])))
            rep_ci["lrc_d"] = ci_d[-1]
            rep_ci["lrc_dz"] = ci_dz[-1]
            if pre_arm:
                bpre = np.asarray(bpre, float)
                se_pre.append(float(np.nanstd(bpre, ddof=1)))
                ci_pre.append(tuple(np.nanpercentile(bpre, [2.5, 97.5])))
                rep_ci["lrc_dz_pre"] = ci_pre[-1]

        if record is not None:
            for k in keys:
                lo, hi = rep_ci.get(k, (None, None))
                record.append(dict(scenario=scenario, dgp=dgp, n_patients=n_patients, rep=rep,
                                   estimator=k, estimate=est[k][-1], ci_lo=lo, ci_hi=hi))

    def summarize(v):
        v = np.asarray(v, dtype=float)
        good = np.isfinite(v)
        v = v[good]
        m = int(v.size)
        if m == 0:
            return dict(bias=float("nan"), esd=float("nan"), rmse=float("nan"),
                        mcse_bias=float("nan"), n=0)
        bias = float(v.mean() - truth)
        esd = float(v.std(ddof=1))
        # the Monte Carlo error of the bias combines replicate scatter with the error in the truth
        return dict(bias=bias, esd=esd,
                    rmse=float(np.sqrt(np.mean((v - truth) ** 2))),
                    mcse_bias=float(np.sqrt(esd ** 2 / m + truth_se ** 2)),
                    mcse_replicates_only=float(esd / np.sqrt(m)), n=m)

    def coverage_block(cis):
        """Binomial coverage MCSE and separate sensitivity to truth approximation."""
        if not cis:
            return {}
        lo = np.asarray([c[0] for c in cis], float)
        hi = np.asarray([c[1] for c in cis], float)
        m = int(lo.size)

        def cov(t):
            return float(np.mean((lo <= t) & (t <= hi)))

        c0, cm, cp = cov(truth), cov(truth - truth_se), cov(truth + truth_se)
        binom = float(np.sqrt(c0 * (1.0 - c0) / m))
        half = float(0.5 * abs(cp - cm))
        return dict(coverage=c0,
                    mcse_coverage=binom,
                    coverage_at_truth_minus_mcse=cm,
                    coverage_at_truth_plus_mcse=cp,
                    coverage_truth_sensitivity_half_range=half)

    res = {k: summarize(v) for k, v in est.items()}
    res["lrc_dz"].update(coverage_block(ci_dz))
    res["lrc_d"].update(coverage_block(ci_d))
    if se_d:
        res["lrc_d"]["mean_se"] = float(np.mean(se_d))
    if se_dz:
        res["lrc_dz"]["mean_se"] = float(np.mean(se_dz))
    if pre_arm:
        res["lrc_dz_pre"].update(coverage_block(ci_pre))
        if se_pre:
            res["lrc_dz_pre"]["mean_se"] = float(np.mean(se_pre))
    for k in ("lrc_d", "lrc_dz", "lrc_dz_pre") if pre_arm else ("lrc_d", "lrc_dz"):
        v = res[k]
        esd = v.get("esd")
        v["se_over_sd"] = (v["mean_se"] / esd) if (v.get("mean_se") and esd) else float("nan")
        v["bias_over_sd"] = (v["bias"] / esd) if esd else float("nan")
    res["_truth"] = truth
    res["_truth_mcse"] = truth_se
    res["_truth_patients"] = int(TRUTH_PATIENTS)
    res["_truth_seed"] = int(TRUTH_SEED[dgp])
    res["_mcse_coverage_note"] = (
        "Coverage MCSE is binomial conditional on the estimated population truth. "
        "The two rescored coverages at truth plus or minus one truth MCSE are a "
        "separate sensitivity analysis; their half-range is not a standard error.")
    res["_support_cost_vs_oracle"] = float(np.nanmean(np.asarray(est["lrc_dz"], float))
                                           - np.nanmean(np.asarray(est["baseline_oracle"], float)))
    if pre_arm:
        res["_support_cost_vs_oracle_pre"] = float(
            np.nanmean(np.asarray(est["lrc_dz_pre"], float))
            - np.nanmean(np.asarray(est["baseline_oracle"], float)))
    res["_diagnostics"] = dict(
        unsupported_share=float(np.nanmean(unsup)),
        fallback_share=float(np.nanmean(fallback)),
        empty_map_cells=float(np.mean(empty)),
        n_patients=n_patients, n_reps=n_reps, n_boot=n_boot,
        n_reps_point=n_reps, n_reps_coverage=n_cov,
    )
    res["_zero_inflation"] = dict(
        zero_share=float(np.mean([z["zero_share"] for z in zdiag])),
        max_duration_with_zero=float(max(z["max_duration_with_zero"] for z in zdiag)),
        mean_max_duration_with_zero=float(np.mean([z["max_duration_with_zero"] for z in zdiag])),
        mixing_duration_cells=float(np.mean([z["mixing_duration_cells"] for z in zdiag])),
        mixing_joint_cells=float(np.mean([z["mixing_joint_cells"] for z in zdiag])),
        share_excursions_in_mixing_duration_cells=float(
            np.mean([z["share_excursions_in_mixing_duration_cells"] for z in zdiag])),
        stochastic_non_annunciation=bool(dgp in ZERO_INFLATED),
    )
    if pre_arm:
        res["_diagnostics"]["fallback_share_prespecified"] = float(np.nanmean(fb_pre))
        res["_diagnostics"]["map_disagreement_share"] = float(np.nanmean(disagree))
    if dgp != scenario:
        res["_diagnostics"]["dgp"] = dgp
    return res


SCENARIO_LABEL = {
    "A": "Transport holds (duration only)",
    "B": "Reward depends on overshoot; composition shifts",
    "C": "As B, sparse feature cells (fallback binds)",
    "D": "Within-pooled-cell heterogeneity",
    "E": ("Partial support failure (baseline CLIPPED at 100 s, giving a point mass there, AND "
          "candidate duration scale raised 11.0 -> 20.0, so the candidate lengthens excursions; the "
          "only scenario where the candidate increases burden)"),
    "F": "Transport violated (reward moves with the limit)",
    "G": "Strong clustering, very unequal excursion counts",
    "H": ("Sample-size sweep of the intended-use setting (scenario A's law at several patient "
          "counts)"),
    "I": ("Zero-inflated: an excursion of any duration can go unannunciated, miss probability "
          "depends on duration only; composition shifts"),
    "J": ("Zero-inflated: miss probability depends on duration AND overshoot; composition shifts"),
}
SCENARIO_PATIENTS = {"A": 300, "B": 300, "C": 60, "D": 300, "E": 300, "F": 300, "G": 300,
                     "I": 300, "J": 300}
ALL_SCENARIOS = "A,B,C,D,E,F,G,I,J"

#: The location of the simulation results, inside the repository and independent of any
#: protected working tree: `run_submission_simulations.py` writes it and `make_sim_table.py` and the
#: simulation figures read it by default. The results are regenerated, not distributed.
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SWEEP_SIZES = (300, 600, 1200, 2400)
REPLICATE_FIELDS = ["scenario", "dgp", "n_patients", "rep", "estimator", "estimate",
                    "ci_lo", "ci_hi"]


def _jsonable(o):
    """Replace every non-finite float by None so the output is valid JSON under a strict parser.

    `json.dump` emits bare `NaN` and `Infinity` tokens by default, which a strict parser rejects.
    Everything is converted here and then written with `allow_nan=False`, which turns any survivor
    into an error rather than an invalid file.
    """
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, (np.floating,)):
        return _jsonable(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(_jsonable(obj), f, indent=1, allow_nan=False)
    return path


def write_replicates(rows, path):
    """Per-replicate point estimates and interval endpoints, plain CSV.

    One row per (scenario, replicate, estimator). `ci_lo`/`ci_hi` are empty when the estimator
    carries no bootstrap interval or the replicate was past the coverage count. This is the file that
    makes a refined truth cheap: rescoring coverage is a filter and two comparisons, not a repeat
    run. Plain, uncompressed CSV: about a megabyte, readable without tooling.
    """
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REPLICATE_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None or (isinstance(r[k], float) and not math.isfinite(r[k]))
                            else (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k]))
                        for k in REPLICATE_FIELDS})
    return path


def run_sweep(sizes, n_reps, n_boot, seed, scenario="A", pre_arm=True, record=None):
    """Scenario H: the intended-use law run at several patient counts.

    The question is whether the coverage of the duration-by-overshoot interval approaches nominal as
    the study grows. Three quantities are recorded at each n:
      coverage            the thing asked about;
      se_over_sd          mean bootstrap standard error over the empirical standard deviation of the
                          estimates, which says whether the interval WIDTH is right;
      bias_over_sd        the centring error in units of that empirical standard deviation.
    `_finding` is written from these numbers by `_sweep_finding` below, never ahead of the run.
    Nothing here is tuned: the sizes, replicate count, bootstrap size and seeds are the ones the main table uses, and the smallest size reproduces the
    main table's scenario A row (same seed, same law, same first `n_reps` replicates).
    """
    out = {"_scenario": "H", "_label": SCENARIO_LABEL["H"], "_dgp": scenario,
           "_settings": dict(sizes=list(sizes), n_reps=n_reps, n_boot=n_boot, seed=seed,
                             prespecified_map_arm=bool(pre_arm),
                             truth_patients=int(TRUTH_PATIENTS),
                             truth_seed=int(TRUTH_SEED[scenario])),
           "sizes": []}
    for n in sizes:
        t0 = time.time()
        r = run_scenario(scenario, n_reps, n, n_boot, seed, pre_arm=pre_arm, record=record)
        if record is not None:
            for row in record:
                if row["scenario"] == scenario and row["n_patients"] == n:
                    row["scenario"] = "H"
        row = dict(n_patients=n, truth=r["_truth"], truth_mcse=r["_truth_mcse"])
        for k in ("lrc_d", "lrc_dz", "lrc_dz_pre") if pre_arm else ("lrc_d", "lrc_dz"):
            v = r[k]
            row[k] = dict(
                bias=v["bias"], esd=v["esd"], rmse=v["rmse"], mcse_bias=v["mcse_bias"],
                mcse_replicates_only=v["mcse_replicates_only"],
                mean_se=v.get("mean_se", float("nan")),
                se_over_sd=v.get("se_over_sd", float("nan")),
                bias_over_sd=v.get("bias_over_sd", float("nan")),
                coverage=v.get("coverage", float("nan")),
                mcse_coverage=v.get("mcse_coverage", float("nan")),
                coverage_truth_sensitivity_half_range=v.get("coverage_truth_sensitivity_half_range", float("nan")),
                coverage_at_truth_minus_mcse=v.get("coverage_at_truth_minus_mcse", float("nan")),
                coverage_at_truth_plus_mcse=v.get("coverage_at_truth_plus_mcse", float("nan")),
            )
        row["_diagnostics"] = r["_diagnostics"]
        out["sizes"].append(row)
        z = row["lrc_dz"]
        print(f"H n={n:5d}  truth={row['truth']:6.2f}%  bias={z['bias']:+6.3f}  "
              f"SD={z['esd']:.3f}  meanSE={z['mean_se']:.3f}  SE/SD={z['se_over_sd']:.3f}  "
              f"bias/SD={z['bias_over_sd']:+.3f}  cov={z['coverage']:.3f}  [{time.time() - t0:.0f}s]")
    out["_finding"] = _sweep_finding(out)
    return out


def _sweep_finding(sweep):
    """Describe the independent-map coverage without a directional template."""
    rows = sweep['sizes']
    pairs = ', '.join(f"{r['n_patients']}: {r['lrc_dz_pre']['coverage']:.3f}" for r in rows)
    return ('Independent-map feature calibration: coverage by patient count is ' + pairs
            + '. These finite-simulation estimates are reported with binomial Monte Carlo uncertainty; '
              'they do not establish monotone convergence or a persistent centring error.')

def main():
    ap = argparse.ArgumentParser()
    # 500 point-estimation replicates (the first 400 also scored for coverage) are the paper's
    # settings, so `make simulation` with no arguments reproduces the reported results.
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--patients", type=int, default=0)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--scenarios", default=ALL_SCENARIOS)
    ap.add_argument("--out", default="", help="output directory (default: simulation/results)")
    ap.add_argument("--sweep", default="", nargs="?", const=",".join(str(n) for n in SWEEP_SIZES),
                    help="run the sample-size sweep (scenario H) at these patient counts instead of "
                         "the scenario table; bare --sweep uses "
                         + ",".join(str(n) for n in SWEEP_SIZES))
    ap.add_argument("--sweep-scenario", default="A",
                    help="which scenario's law the sweep runs (default A, the intended-use setting)")
    ap.add_argument("--no-prespecified-map", action="store_true",
                    help="drop the prespecified-map arm (it costs one extra sample per replicate)")
    ap.add_argument("--verify-truth-vectorization", default="",
                    help="run the vectorized and the pre-vectorization truth side by side over many "
                         "seeds for these scenarios and print the comparison, then exit")
    ap.add_argument("--verify-seeds", type=int, default=30)
    args = ap.parse_args()
    pre_arm = not args.no_prespecified_map

    if args.verify_truth_vectorization:
        scs = [s.strip() for s in args.verify_truth_vectorization.split(",") if s.strip()]
        verify_truth_vectorization(scs, n_seeds=args.verify_seeds)
        return

    dest = args.out or RESULTS
    os.makedirs(dest, exist_ok=True)

    if args.sweep:
        sizes = [int(x) for x in args.sweep.split(",") if x.strip()]
        rows = []
        sweep = run_sweep(sizes, args.reps, args.boot, SEED, args.sweep_scenario, pre_arm,
                          record=rows)
        path = write_json(sweep, os.path.join(dest, "SIMULATION_SAMPLE_SIZE.json"))
        rpath = write_replicates(rows, os.path.join(dest,
                                                    "SIMULATION_SAMPLE_SIZE_REPLICATES.csv"))
        print(sweep["_finding"])
        print("wrote", path)
        print("wrote", rpath, f"({len(rows)} replicate rows)")
        return

    out = {}
    rows = []
    for sc in args.scenarios.split(","):
        sc = sc.strip()
        if not sc:
            continue
        npat = args.patients or SCENARIO_PATIENTS.get(sc, 300)
        t0 = time.time()
        out[sc] = run_scenario(sc, args.reps, npat, args.boot, SEED, pre_arm=pre_arm, record=rows)
        out[sc]["_label"] = SCENARIO_LABEL[sc]
        r = out[sc]
        pre = r.get("lrc_dz_pre")
        pre_txt = (f"  pre={pre['bias']:+6.2f}" if pre else "")
        cov_pre = (f"  cov(pre)={pre.get('coverage', float('nan')):.3f}" if pre else "")
        print(f"{sc} truth={r['_truth']:7.3f}%  "
              f"replay={r['replay']['bias']:+6.2f}  LRC-d={r['lrc_d']['bias']:+6.2f}  "
              f"LRC-dz={r['lrc_dz']['bias']:+6.2f}{pre_txt}  "
              f"b-oracle={r['baseline_oracle']['bias']:+6.2f}  "
              f"cov(d)={r['lrc_d'].get('coverage', float('nan')):.3f}  "
              f"cov(dz)={r['lrc_dz'].get('coverage', float('nan')):.3f}{cov_pre}  "
              f"[{time.time() - t0:.0f}s]")

    if "B" in out and "C" in out:
        # B and C share a law exactly, so their two INDEPENDENTLY seeded truth estimates are a
        # consistency check on the truth computation itself.
        db = out["B"]["_truth"] - out["C"]["_truth"]
        sd = (out["B"]["_truth_mcse"] ** 2 + out["C"]["_truth_mcse"] ** 2) ** 0.5
        print(f"truth cross-check: B and C share a law; independently seeded truths differ by "
              f"{db:+.4f} points against a combined Monte Carlo standard error of {sd:.4f} "
              f"(z = {db / sd:+.2f})")

    path = write_json(out, os.path.join(dest, "SIMULATION_LINKED_REWARD.json"))
    rpath = write_replicates(rows, os.path.join(dest, "SIMULATION_LINKED_REWARD_REPLICATES.csv"))
    print("wrote", path)
    print("wrote", rpath, f"({len(rows)} replicate rows)")


if __name__ == "__main__":
    main()
