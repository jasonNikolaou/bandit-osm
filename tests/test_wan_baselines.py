"""Checks of the two Wan algorithms' mathematical specializations."""
import math

import numpy as np

from algorithms.wan_baselines import (
    ALPHA, BanditMLSM, PoissonBE, _barrier_direction,
    _barrier_hessian_times, _capped_entropy_projection,
    _simplex_barrier_ftrl,
)


def test_product_simplex_barrier_direction_and_ftrl_optimality():
    rng = np.random.default_rng(10)
    x = np.array([[.12, .37, .08], [.30, .05, .15]])
    for _ in range(20):
        h = _barrier_direction(x, rng)
        assert np.isclose(np.sum(h * _barrier_hessian_times(x, h)), 1)
        assert np.all(x + h >= -1e-13)
        assert np.all((x + h).sum(axis=1) <= 1 + 1e-13)
    gradient = np.array([[.6, -.2, 1.5], [-.7, 3.0, 1.2]])
    eta = .07
    chosen = _simplex_barrier_ftrl(gradient, eta)
    slack = 1 - chosen.sum(axis=1)
    residual = -1 / chosen + 1 / slack[:, None] - eta * gradient
    assert np.max(np.abs(residual)) < 1e-9


def test_2023_exploration_scalar_is_unbiased_for_directional_derivative():
    # A multilinear extension on one simplex; exact averaging over every
    # coordinate and both coin outcomes, rather than copying update code.
    x = np.array([.13, .21, .07])
    h = np.array([.03, -.04, .02])
    d = len(x)

    def f(y):
        return .2 * y[0] + .4 * y[1] + .1 * y[2] + .3 * y[0] * y[2]

    gradient = np.array([.2 + .3 * x[2], .4, .1 + .3 * x[0]])
    for z in (.3, .8):
        base = z * x
        if z >= .5:
            expectation = -ALPHA * d / z * f(base)
            expectation += sum(ALPHA / z * f(base + z * h[i] * np.eye(d)[i])
                               for i in range(d))
        else:
            expectation = sum(
                2 * ALPHA * h[i] * (f(base + .5 * np.eye(d)[i]) - f(base))
                for i in range(d))
        target = ALPHA * np.dot(h, np.array([
            .2 + .3 * base[2], .4, .1 + .3 * base[0]]))
        assert np.isclose(expectation, target)


def test_uniform_matroid_balanced_exchange_marginals():
    # For A={0,2}, the paper requires row sums 1, column sums x, and
    # support only on feasible exchanges.
    x = np.array([.7, .4, .8, .1])
    base = np.array([0, 2])
    q = np.zeros((2, 4))
    outside_mass = x[[1, 3]].sum()
    for row, i in enumerate(base):
        q[row, i] = x[i]
        for j in (1, 3):
            q[row, j] = (1 - x[i]) * x[j] / outside_mass
    assert np.allclose(q.sum(axis=1), 1)
    assert np.allclose(q.sum(axis=0), x)
    assert q[0, 2] == q[1, 0] == 0


def test_2026_one_query_statistic_estimates_insertion_loss():
    # Condition on an exchange and event time. Enumerate both possible
    # leave-one-out sets and both coin outcomes for a non-additive reward.
    tau = .37
    probabilities = [.2, .5, .4]

    def f(subset):
        return 1 - math.prod(1 - probabilities[item] for item in subset)

    insertion = ((1 - tau) * (f({2}) - f(set()))
                 + tau * (f({1, 2}) - f({1})))
    expected_statistic = 0
    for retained, chance in ((set(), 1 - tau), ({1}, tau)):
        expected_statistic += chance * (
            .5 * 2 * (1 - f(retained | {2}))
            + .5 * 2 * f(retained))
    assert np.isclose(expected_statistic, 1 - insertion)


def test_capped_entropy_projection_satisfies_kkt_conditions():
    weights = np.array([.5, .05, .3, .15])
    lower, upper = .03, .7
    result = _capped_entropy_projection(np.log(weights), 2, lower, upper)
    assert np.isclose(result.sum(), 2)
    assert np.all(result >= lower - 1e-12)
    assert np.all(result <= upper + 1e-12)
    free = (result > lower + 1e-8) & (result < upper - 1e-8)
    assert np.ptp(np.log(result[free] / weights[free])) < 1e-10


def test_both_wan_policies_play_feasible_sets_with_one_feedback():
    for cls in (BanditMLSM, PoissonBE):
        policy = cls(8, 3, 75, seed=123)
        for _ in range(75):
            action = policy.act()
            assert len(action) <= 3
            assert len(set(action)) == len(action)
            assert np.all((0 <= action) & (action < 8))
            policy.observe(len(action) / 3)
        assert np.all(np.isfinite(policy.iterate()))
        if cls is PoissonBE:
            assert np.isclose(policy.iterate().sum(), 3)
        else:
            assert policy.diagnostics()["exploration_rounds"] == math.ceil(75 / 32)
