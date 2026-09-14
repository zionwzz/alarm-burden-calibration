#!/usr/bin/env python3
"""Synthetic tests for `alarmreplay.linked_reward`. No data and no network required.

The cases are the ones that break a cell estimator: all-zero rewards, cells whose true reward is
zero, several alarm runs on one excursion, exact against pooled carriers, target cells the baseline
does not support, coarsened cells, and zero denominators. One test checks the algebraic identity that
lets the pooled tail cells use a seconds carrier while the exact cells use counts.

Four of the tests pin down conventions the application relies on: the high-duration plateau is the
tail-inclusive pooled ratio; the support condition of the duration-by-feature estimand is on joint
cells, not on the duration marginal; the coarsening is reported by the burden it carries and not by
a count of cells; and the serialized feature map round trips exactly, so a later prespecified run
evaluates the same estimator.

Usage: python3 tools/test_linked_reward.py
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alarmreplay.linked_reward import (  # noqa: E402
    COUNT, PLATEAU_FROM_INDEX, SECONDS, CellMap, burden_contrast, estimate, fallback_diagnostics,
    feature_support_map, joint_carrier, joint_support_diagnostics, joint_target_seconds, kappa,
    plateau_ratio, replayed_burden, reward_rates,
    standardized_burden, support_diagnostics,
)

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILS.append(name)


def approx(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


# --------------------------------------------------------------------------------- the cell map
def test_cellmap():
    print("cell map")
    m = CellMap()
    check("cell count", m.n_cells == 64)
    check("exact cells carry counts", set(m.carrier[: m.n_exact]) == {COUNT})
    check("pooled cells carry seconds", set(m.carrier[m.n_exact:]) == {SECONDS})
    idx = m.index([0.0, 1.9, 2.0, 118.0, 119.9, 120.0, 179.0, 180.0, 299.0, 300.0, 599.0, 600.0, 5000.0])
    check("exact indexing", list(idx[:5]) == [0, 0, 1, 59, 59])
    check("tail indexing", list(idx[5:]) == [60, 60, 61, 61, 62, 62, 63, 63])
    d = m.representative_duration
    check("exact cell durations are the grid", approx(d[1], 2.0) and approx(d[59], 118.0))
    check("every exact cell holds one duration on a 2 s grid",
          np.allclose(np.diff(d[: m.n_exact]), m.width))


# ------------------------------------------------------------------------------- reward rates
def test_reward_rates():
    print("reward rates")
    S = np.array([0.0, 10.0, 30.0, 0.0])
    R = np.array([5.0, 5.0, 10.0, 0.0])
    beta = reward_rates(S, R)
    check("rate is linked seconds per carrier unit", np.allclose(beta[:3], [0.0, 2.0, 3.0]))
    check("cell with no carrier mass contributes nothing", beta[3] == 0.0)

    check("all-zero rewards give a zero burden",
          standardized_burden(reward_rates(np.zeros(4), R), np.ones(4)) == 0.0)

    fb = np.array([1.0, 1.0, 1.0, 1.0])
    beta_fb = reward_rates(S, np.array([5.0, 5.0, 0.0, 0.0]), fallback=fb)
    check("fallback fills only the empty cells", np.allclose(beta_fb, [0.0, 2.0, 1.0, 1.0]))

    S2 = np.array([[0.0, 4.0], [6.0, 0.0]])
    R2 = np.array([[2.0, 2.0], [3.0, 0.0]])
    beta2 = reward_rates(S2, R2, fallback=np.array([9.0, 9.0]))
    check("fallback broadcasts over the feature axis",
          np.allclose(beta2, [[0.0, 2.0], [2.0, 9.0]]))
    check("a true zero reward stays zero and is not treated as missing", beta2[0, 0] == 0.0)


# ------------------------------------------------------------------- carriers and the identity
def test_carrier_identity():
    """The pooled-tail seconds carrier reproduces the count carrier scaled by the mean duration.

    In an exact cell the estimator uses beta = S/N and multiplies by the target count. In a pooled
    cell it uses beta = S/D and multiplies by the target seconds. The two agree whenever the pooled
    cell's mean duration is carried across with the count, which is the algebra the application's
    tail handling performs; this test pins it down.
    """
    print("carriers")
    S_c, N_c, D_c = 240.0, 40.0, 6000.0          # one pooled cell at the limit in force
    n_t, secs_t = 25.0, 4000.0                   # the same cell under the candidate limit

    beta_seconds = S_c / D_c                     # annunciation ratio, dimensionless
    burden_seconds_carrier = beta_seconds * secs_t

    beta_count = S_c / N_c                       # linked seconds per excursion
    dbar_base = D_c / N_c                        # mean duration in the cell at the limit in force
    dbar_target = secs_t / n_t                   # mean duration in the cell under the candidate
    burden_count_carrier = n_t * beta_count * dbar_target / dbar_base

    check("seconds carrier equals the rescaled count carrier",
          approx(burden_seconds_carrier, burden_count_carrier))
    check("both are linked seconds", approx(burden_seconds_carrier, 160.0))

    m = CellMap(width=2.0, n_exact=3, tail_edges=(6.0,), tail_midpoints=(9.0,))
    counts = np.array([10.0, 5.0, 2.0, 4.0])
    tail_secs = np.array([36.0])
    check("replayed burden sums exact count x duration plus pooled seconds",
          approx(replayed_burden(m, counts, tail_secs), 0 * 10 + 2 * 5 + 4 * 2 + 36.0))


# ----------------------------------------------------------------- standardization and contrasts
def test_standardization():
    print("standardization")
    m = CellMap(width=2.0, n_exact=3, tail_edges=(6.0,), tail_midpoints=(9.0,))
    S = np.array([0.0, 5.0, 12.0, 60.0])
    R = np.array([10.0, 5.0, 4.0, 120.0])        # counts, counts, counts, seconds
    beta = reward_rates(S, R)
    check("beta", np.allclose(beta, [0.0, 1.0, 3.0, 0.5]))

    R_base = R
    R_tgt = np.array([20.0, 4.0, 2.0, 60.0])
    b0 = standardized_burden(beta, R_base)
    b1 = standardized_burden(beta, R_tgt)
    check("baseline burden reproduces the observed linked seconds", approx(b0, S.sum()))
    check("target burden", approx(b1, 0 * 20 + 1 * 4 + 3 * 2 + 0.5 * 60))

    check("percentage change", approx(burden_contrast(b1, b0), 100 * (1 - 40.0 / 77.0)))
    check("no change when the target equals the baseline", approx(burden_contrast(b0, b0), 0.0))

    rep0 = replayed_burden(m, R_base[:3], R_base[3:])
    rep1 = replayed_burden(m, R_tgt[:3], R_tgt[3:])
    k = kappa(b0, b1, rep0, rep1)
    check("kappa is a ratio of ratios",
          approx(k, (b0 / rep0) / (b1 / rep1)))
    check("kappa is one when the reward is proportional to duration",
          approx(kappa(10.0, 5.0, 20.0, 10.0), 1.0))


# --------------------------------------------------------------------------- degenerate inputs
def test_degenerate():
    print("degenerate inputs")
    check("zero replayed burden gives nan, not an exception",
          np.isnan(kappa(1.0, 1.0, 0.0, 1.0)))
    check("zero target burden gives nan", np.isnan(kappa(1.0, 0.0, 1.0, 0.0)))
    check("zero baseline burden in a contrast gives nan", np.isnan(burden_contrast(1.0, 0.0)))
    check("every output is finite or explicitly nan",
          np.isfinite(kappa(2.0, 1.0, 4.0, 2.0)))
    try:
        reward_rates(np.zeros(3), np.zeros(4))
        check("shape mismatch raises", False)
    except ValueError:
        check("shape mismatch raises", True)


# ------------------------------------------------------------------------ multiple alarm runs
def test_multiple_runs():
    """Y is total attached seconds, so several runs on one excursion need no extra assumption."""
    print("multiple alarm runs")
    m = CellMap(width=2.0, n_exact=2, tail_edges=(4.0,), tail_midpoints=(6.0,))
    # cell 1 holds three excursions: one with two runs (3 s + 4 s), one with one run (5 s), one unlinked
    S = np.array([0.0, 3.0 + 4.0 + 5.0 + 0.0, 0.0])
    R = np.array([0.0, 3.0, 0.0])
    beta = reward_rates(S, R)
    check("total attached seconds per excursion", approx(beta[1], 12.0 / 3.0))
    check("the unlinked excursion is in the denominator", approx(beta[1], 4.0))
    S_one_run = np.array([0.0, 5.0, 0.0])
    check("dropping the multi-run excursion's second run lowers the rate",
          reward_rates(S_one_run, R)[1] < beta[1])


# ------------------------------------------------------------------ support and fallback cells
def test_support_and_fallback():
    print("support and fallback")
    m = CellMap(width=2.0, n_exact=3, tail_edges=(6.0,), tail_midpoints=(9.0,))
    baseline_counts = np.array([50.0, 0.0, 30.0, 5.0])       # cell 1 has no baseline excursion
    target_counts = np.array([10.0, 10.0, 10.0, 3.0])
    target_tail = np.array([27.0])
    d = support_diagnostics(m, baseline_counts, target_counts, target_tail, min_for_precision=20)
    secs = 0 * 10 + 2 * 10 + 4 * 10 + 27.0
    check("replayed seconds", approx(d["replayed_seconds"], secs))
    check("unsupported share is the target mass in empty baseline cells",
          approx(d["share_unsupported"], 20.0 / secs))
    check("imprecise share separates precision from identification",
          approx(d["share_imprecise"], (20.0 + 27.0) / secs))
    check("pooled tail share", approx(d["share_pooled_tail"], 27.0 / secs))

    counts2 = np.array([[40.0, 3.0], [2.0, 1.0]])
    ok = feature_support_map(counts2, nmin=5)
    check("fallback map is a threshold on baseline counts",
          ok.tolist() == [[True, False], [False, False]])

    fit = estimate(
        CellMap(width=2.0, n_exact=2, tail_edges=(4.0,), tail_midpoints=(6.0,)),
        linked_seconds=np.array([0.0, 20.0, 30.0]),
        carrier_totals=np.array([10.0, 10.0, 60.0]),
        targets={"u1": np.array([5.0, 5.0, 30.0])},
        linked_seconds_feature=np.array([[0.0, 0.0], [18.0, 2.0], [30.0, 0.0]]),
        carrier_totals_feature=np.array([[8.0, 2.0], [9.0, 1.0], [60.0, 0.0]]),
        targets_feature={"u1": np.array([[4.0, 1.0], [4.0, 1.0], [30.0, 0.0]])},
        feature_ok=np.array([[True, False], [True, False], [True, False]]),
    )
    check("fallback cells are counted", fit.fallback_cells == 3)
    check("a cell outside the map takes the duration-only reward",
          approx(fit.beta_feature[0, 1], fit.beta_duration[0]))
    check("a cell inside the map keeps its own reward",
          approx(fit.beta_feature[1, 0], 2.0))
    check("feature burden is finite", np.isfinite(fit.burden["feature"]["u1"]))
    check("duration burden is finite", np.isfinite(fit.burden["duration"]["u1"]))

    fit2 = estimate(
        CellMap(width=2.0, n_exact=2, tail_edges=(4.0,), tail_midpoints=(6.0,)),
        linked_seconds=np.array([0.0, 20.0, 30.0]),
        carrier_totals=np.array([10.0, 10.0, 60.0]),
        targets={"u1": np.array([5.0, 5.0, 30.0])},
        linked_seconds_feature=np.array([[0.0, 0.0], [18.0, 2.0], [30.0, 0.0]]),
        carrier_totals_feature=np.array([[8.0, 2.0], [9.0, 0.0], [60.0, 0.0]]),
        targets_feature={"u1": np.array([[4.0, 1.0], [4.0, 1.0], [30.0, 0.0]])},
        feature_ok=np.array([[True, True], [True, True], [True, True]]),
    )
    check("a map cell empty in this sample is counted, not silently dropped",
          fit2.empty_map_cells == 2)


# --------------------------------------------------------- equivalence with the application form
def test_equivalence_with_application_form():
    """The module reproduces the algebra `analysis/link_analysis.py` computes.

    The application forms the duration-only linked burden as ``h @ ps``, where ``h`` are the target
    counts by cell and ``ps`` is the per-excursion reward with the pooled cells rescaled by the ratio
    of the target cell's mean duration to the baseline cell's. That expression equals the generic
    carrier form with counts in exact cells and seconds in pooled cells.
    """
    print("equivalence with the implemented estimator")
    rng = np.random.default_rng(11)
    m = CellMap(width=2.0, n_exact=6, tail_edges=(12.0, 20.0), tail_midpoints=(16.0, 30.0))
    nc, ne = m.n_cells, m.n_exact

    N = rng.integers(1, 40, nc).astype(float)                        # baseline counts by cell
    Dtot = np.concatenate([N[:ne] * m.representative_duration[:ne],
                           N[ne:] * rng.uniform(13.0, 28.0, nc - ne)])
    S = Dtot * rng.uniform(0.0, 0.9, nc)
    S[2] = 0.0                                                       # a cell whose reward is zero
    h = rng.integers(0, 30, nc).astype(float)                        # target counts by cell
    tail_secs = h[ne:] * rng.uniform(13.0, 28.0, nc - ne)

    # ---- the application's expression
    psi = np.divide(S, N, out=np.zeros_like(S), where=N > 0)
    dmean = np.divide(Dtot, N, out=m.representative_duration.copy(), where=N > 0)
    ps = psi.copy()
    for b in range(m.n_tail):
        c = ne + b
        if h[c] > 0 and dmean[c] > 0:
            ps[c] = psi[c] * (tail_secs[b] / h[c]) / dmean[c]
    applied = float(h @ ps)

    # ---- the generic carrier form
    carrier_base = np.concatenate([N[:ne], Dtot[ne:]])
    carrier_target = np.concatenate([h[:ne], tail_secs])
    beta = reward_rates(S, carrier_base)
    module = standardized_burden(beta, carrier_target)

    check("generic carrier form reproduces the implemented burden", approx(module, applied, 1e-10),
          f"{module!r} vs {applied!r}")
    check("both are positive and finite", np.isfinite(module) and module > 0)

    rep_applied = float(h[:ne] @ m.representative_duration[:ne] + tail_secs.sum())
    check("replayed burden agrees", approx(replayed_burden(m, h[:ne], tail_secs), rep_applied))


# -------------------------------------------------------------------- the plateau, one definition
def test_plateau_definition():
    """The plateau is the pooled ratio from cell 30 to the END of the map, tail cells included.

    The array below is built so the two candidate conventions cannot be confused: every exact cell
    above the cut is annunciated half the time, and the four pooled tail cells hold four times that
    much replayed mass and no linked mass at all. Dropping the tail would report 0.5; keeping it
    reports 30/220. The test asserts the tail-inclusive value, because the tail cells hold the
    longest excursions and excluding them silently narrows the estimand while keeping its name.
    """
    print("plateau definition")
    m = CellMap()
    check("the default cut is cell 30", PLATEAU_FROM_INDEX == 30)
    check("cell 30 is 60 s on the exact grid",
          approx(m.representative_duration[PLATEAU_FROM_INDEX], 60.0))

    linked = np.zeros(m.n_cells)
    total = np.zeros(m.n_cells)
    linked[:30] = 9.0                                  # mass below the cut, which must not count
    total[:30] = 11.0
    linked[30:m.n_exact] = 1.0                         # 30 exact cells: 1 linked of 2
    total[30:m.n_exact] = 2.0
    linked[m.n_exact:] = 0.0                           # 4 pooled tail cells: none linked...
    total[m.n_exact:] = 40.0                           # ...and a lot of mass

    tail_inclusive = 30.0 / 220.0
    tail_exclusive = 30.0 / 60.0
    check("the two conventions really do differ here", not approx(tail_inclusive, tail_exclusive))
    check("plateau_ratio is the tail-INCLUSIVE pooled ratio",
          approx(plateau_ratio(linked, total), tail_inclusive),
          f"{plateau_ratio(linked, total)!r}")
    check("it is not the tail-exclusive one",
          not approx(plateau_ratio(linked, total), tail_exclusive))
    check("the tail-exclusive slice is reachable only by truncating the input",
          approx(plateau_ratio(linked[: m.n_exact], total[: m.n_exact]), tail_exclusive))

    linked2, total2 = linked.copy(), total.copy()
    linked2[:30] = 0.0
    total2[:30] = 1000.0
    check("mass below the cut never enters", approx(plateau_ratio(linked2, total2), tail_inclusive))

    check("a pooled ratio weights cells by their mass, unlike a mean of cell shares",
          not approx(tail_inclusive, float(np.mean(np.divide(linked[30:], total[30:])))))
    check("an empty slice gives nan, not a division error",
          np.isnan(plateau_ratio(np.zeros(64), np.zeros(64))))
    check("the cut is a parameter, not a constant",
          approx(plateau_ratio(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]), from_index=1),
                 5.0 / 10.0))


# ------------------------------------------------------ support on the joint cells of the estimand
def _joint_fixture():
    """Four duration cells (three exact on a 2 s grid, one pooled) x two feature bins."""
    m = CellMap(width=2.0, n_exact=3, tail_edges=(6.0,), tail_midpoints=(9.0,))
    baseline = np.array([[10.0, 0.0],                  # feature bin 1 is empty at every duration...
                         [8.0, 0.0],
                         [6.0, 3.0],                   # ...except here, where it is sparse
                         [5.0, 0.0]])
    tgt_counts = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    tgt_seconds = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [40.0, 20.0]])
    return m, baseline, tgt_counts, tgt_seconds


def test_joint_support_diagnostics():
    """A target fully supported on the duration marginal can be unsupported on the joint cells."""
    print("joint-cell support")
    m, baseline, tgt_counts, tgt_seconds = _joint_fixture()

    secs = joint_target_seconds(m, tgt_counts, tgt_seconds)
    check("exact joint cells carry count x the cell's one duration",
          np.allclose(secs[:3], [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]]))
    check("pooled joint cells carry their recorded seconds",
          np.allclose(secs[3], [40.0, 20.0]))
    check("the carrier is counts in exact cells and seconds in pooled ones",
          np.allclose(joint_carrier(m, tgt_counts, tgt_seconds),
                      [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [40.0, 20.0]]))

    d = joint_support_diagnostics(m, baseline, tgt_counts, tgt_seconds, nmin=5)
    total = 72.0
    check("joint replayed seconds", approx(d["replayed_seconds_joint"], total))
    check("unsupported joint share is target seconds in joint cells with zero baseline",
          approx(d["share_unsupported_joint"], (0.0 + 2.0 + 20.0) / total))
    check("sparse joint share is 1..nmin-1, and excludes the cell holding exactly nmin",
          approx(d["share_sparse_joint"], 4.0 / total))
    check("the two shares do not overlap",
          d["share_unsupported_joint"] + d["share_sparse_joint"] < 1.0)

    # the duration marginal says the very same target is fully supported
    marg_baseline = baseline.sum(1)
    marg = support_diagnostics(m, marg_baseline, tgt_counts.sum(1), tgt_seconds[m.n_exact:].sum(1))
    check("every duration cell holds baseline excursions", bool((marg_baseline > 0).all()))
    check("the duration marginal reports zero unsupported share", marg["share_unsupported"] == 0.0)
    check("while the joint cells report a positive one", d["share_unsupported_joint"] > 0.0,
          f"{d['share_unsupported_joint']!r}")
    check("so the duration answer must not be quoted as the estimand's support",
          not approx(marg["share_unsupported"], d["share_unsupported_joint"]))

    d10 = joint_support_diagnostics(m, baseline, tgt_counts, tgt_seconds, nmin=10)
    check("raising nmin moves mass from supported to sparse",
          d10["share_sparse_joint"] > d["share_sparse_joint"]
          and approx(d10["share_unsupported_joint"], d["share_unsupported_joint"]))
    empty = joint_support_diagnostics(m, baseline, np.zeros((4, 2)), np.zeros((4, 2)))
    check("a target with no mass gives nan, not a division error",
          np.isnan(empty["share_unsupported_joint"]))


# ------------------------------------------------------------- the fallback, weighted by burden
def test_fallback_burden_diagnostics():
    """A count of fallback cells says nothing about what they do to the estimate."""
    print("burden-weighted fallback")
    m, baseline, tgt_counts, tgt_seconds = _joint_fixture()
    uses_own_rate = feature_support_map(baseline, nmin=5)
    check("the coarsened set is the complement of the prespecified map",
          uses_own_rate.tolist() == [[True, False], [True, False], [True, False], [True, False]])

    beta = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 1.0], [0.5, 0.25]])
    d = fallback_diagnostics(m, uses_own_rate, beta, tgt_counts, tgt_seconds)

    check("cells routed through the fallback are still counted", d["n_cells_fallback"] == 4)
    check("share of the target's replayed seconds on the fallback rate",
          approx(d["share_target_seconds_fallback"], 26.0 / 72.0),
          f"{d['share_target_seconds_fallback']!r}")
    check("share of the standardized linked burden the fallback contributes",
          approx(d["share_linked_burden_fallback"], 20.0 / 46.0),
          f"{d['share_linked_burden_fallback']!r}")
    check("the burden share is not the share of cells",
          not approx(d["share_linked_burden_fallback"], 4.0 / 8.0))
    check("nor is the seconds share",
          not approx(d["share_target_seconds_fallback"], 4.0 / 8.0))
    check("and the two burden-weighted shares are not each other either",
          not approx(d["share_linked_burden_fallback"], d["share_target_seconds_fallback"]))

    full = fallback_diagnostics(m, np.ones_like(uses_own_rate), beta, tgt_counts, tgt_seconds)
    check("a map that keeps every cell has no fallback burden",
          full["share_linked_burden_fallback"] == 0.0
          and full["share_target_seconds_fallback"] == 0.0)
    none = fallback_diagnostics(m, np.zeros_like(uses_own_rate), beta, tgt_counts, tgt_seconds)
    check("a map that keeps no cell routes everything through the fallback",
          approx(none["share_linked_burden_fallback"], 1.0)
          and approx(none["share_target_seconds_fallback"], 1.0))

    # the whole point: the same cell count with a heavier target changes the burden it carries
    heavier = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [40.0, 400.0]])
    d2 = fallback_diagnostics(m, uses_own_rate, beta, tgt_counts, heavier)
    check("same fallback cells, different weight, different answer",
          d2["n_cells_fallback"] == d["n_cells_fallback"]
          and d2["share_linked_burden_fallback"] > d["share_linked_burden_fallback"])


# ------------------------------------------------------------- the prespecified map survives a round trip
def test_feature_map_round_trip():
    """Serializing the support map and loading it back must not change a single cell or estimate.

    `analysis/link_analysis.py` writes the map it used to FEATURE_MAP.json so a later run can apply it
    unchanged (ALARM_FEATURE_MAP), and `analysis/transport_diagnostics.py` reads the same file. If the
    round trip lost or flipped a cell, the later run would silently evaluate a different
    estimator; here the reloaded map is checked cell by cell and the estimates are recomputed with it.
    """
    print("feature map round trip")
    rng = np.random.default_rng(5)
    m = CellMap(width=2.0, n_exact=4, tail_edges=(8.0, 14.0), tail_midpoints=(11.0, 18.0))
    nc, nf = m.n_cells, 3
    counts2 = rng.integers(0, 12, (nc, nf)).astype(float)
    ok = feature_support_map(counts2, nmin=5)
    check("the map has both kinds of cell", bool(ok.any() and not ok.all()))

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "FEATURE_MAP.json")
        with open(path, "w") as f:
            json.dump({"ch": ok.tolist()}, f, indent=0)
        with open(path) as f:
            back = np.asarray(json.load(f)["ch"], dtype=bool)

        check("the reloaded map has the same shape", back.shape == ok.shape)
        check("every cell survives the round trip", np.array_equal(back, ok))
        check("the retained-cell count matches", int(back.sum()) == int(ok.sum()))

        s2 = counts2 * rng.uniform(0.0, 3.0, (nc, nf))
        s1 = s2.sum(1)
        n1 = counts2.sum(1)
        tgt = rng.integers(0, 9, (nc, nf)).astype(float)
        kw = dict(linked_seconds=s1, carrier_totals=n1, targets={"u1": n1},
                  linked_seconds_feature=s2, carrier_totals_feature=counts2,
                  targets_feature={"u1": tgt})
        before = estimate(m, feature_ok=ok, **kw)
        after = estimate(m, feature_ok=back, **kw)
        check("the standardized burden is unchanged by the round trip",
              approx(after.burden["feature"]["u1"], before.burden["feature"]["u1"], 1e-12),
              f"{after.burden['feature']['u1']!r} vs {before.burden['feature']['u1']!r}")
        check("so are the reward rates", np.array_equal(after.beta_feature, before.beta_feature))
        check("and the fallback count", after.fallback_cells == before.fallback_cells)

        cm_path = os.path.join(tmp, "CELL_MAP.json")
        m.to_json(cm_path)
        m2 = CellMap.from_json(cm_path)
        check("the cell map round trips too", m2 == m)
        check("and indexes durations identically",
              np.array_equal(m2.index([0.0, 3.0, 9.0, 40.0]), m.index([0.0, 3.0, 9.0, 40.0])))


# ------------------------------------------------------------------------------ cluster bootstrap
def test_cluster_bootstrap():
    print("patient-cluster resampling")
    from alarmreplay.linked_reward import cluster_bootstrap, percentile_ci
    n = 200
    rng = np.random.default_rng(3)
    S_i = rng.gamma(2.0, 5.0, n)
    R_i = rng.gamma(4.0, 3.0, n)

    def stat(idx):
        return {"ratio": float(S_i[idx].sum() / R_i[idx].sum())}

    draws = cluster_bootstrap(n, stat, n_replicates=300, seed=7)
    ci = percentile_ci(draws)
    point = S_i.sum() / R_i.sum()
    check("interval covers the point estimate", ci["ratio"][0] <= point <= ci["ratio"][1])
    check("interval is not degenerate", ci["ratio"][1] > ci["ratio"][0])
    check("the same seed gives the same draws",
          np.allclose(draws["ratio"], cluster_bootstrap(n, stat, 300, seed=7)["ratio"]))


if __name__ == "__main__":
    for t in (test_cellmap, test_reward_rates, test_carrier_identity, test_standardization,
              test_degenerate, test_multiple_runs, test_support_and_fallback,
              test_plateau_definition, test_joint_support_diagnostics,
              test_fallback_burden_diagnostics, test_feature_map_round_trip,
              test_equivalence_with_application_form, test_cluster_bootstrap):
        t()
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {FAILS}")
        sys.exit(1)
    print("all linked-reward tests passed")
