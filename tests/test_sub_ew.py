import itertools
import math

import numpy as np
import pytest

from algorithms.sub_ew import (SubEW, minimum_variance_coefficients,
                                   sampling_coefficients, shifted_feedback)


@pytest.mark.parametrize("n,k", [(4, 2), (7, 3), (100, 10)])
def test_coefficients_and_approximation_margin(n, k):
    rho, beta, q, b, a, variance = sampling_coefficients(n, k)
    assert beta == pytest.approx(1 / math.e - rho ** k)
    assert np.all(q > 0) and q.sum() == pytest.approx(1)
    expected = np.array([
        sum(rho ** (ell - m) * q[ell - 1] for ell in range(m, k + 1))
        for m in range(1, k + 1)
    ])
    np.testing.assert_allclose(b, expected)
    np.testing.assert_allclose(a, b / np.arange(1, k + 1))
    alpha = sum(q[m - 1] * (1 - rho ** m) for m in range(1, k + 1))
    assert alpha >= 1 - 1 / math.e + beta * rho ** k - 1e-12
    assert variance == pytest.approx(
        sum(b[m - 1] ** 2 * (n + m - 1) / (m * q[m - 1])
            for m in range(1, k + 1)))


def test_minimum_variance_q_for_influence():
    classic = sampling_coefficients(34, 2)
    rho, _, q, _, _, variance = minimum_variance_coefficients(34, 2)
    # For k=2, alpha(q)=3/4-q_1/4; V_q decreases across the feasible interval.
    boundary = 4 * (.75 - (1 - 1 / math.e))
    assert q[0] == pytest.approx(boundary, abs=1e-8)
    assert np.dot(q, 1 - rho ** np.arange(1, 3)) == pytest.approx(
        1 - 1 / math.e)
    assert variance < classic[5]


def test_minimum_variance_q_for_three_draws():
    rho, _, q, _, _, variance = minimum_variance_coefficients(100, 3)
    assert np.all(q > 0)
    assert q.sum() == pytest.approx(1)
    assert np.dot(q, 1 - rho ** np.arange(1, 4)) >= 1 - 1 / math.e - 1e-9
    assert variance < sampling_coefficients(100, 3)[5]


def test_custom_q_endpoints():
    for q, expected_alpha, expected_variance in [
        ([0, 1], .75, None), ([1, 0], .5, 34.0)
    ]:
        policy = SubEW(34, 2, 10, 5, q_mode="custom", q=q, eta=.008)
        assert policy.parameters["alpha_q"] == pytest.approx(expected_alpha)
        assert policy.parameters["V_q"] == expected_variance
        for _ in range(10):
            policy.act()
            policy.observe(.5)
        assert policy.diagnostics()["sampling_cache_underflow_rounds"] == 0


def test_estimator_expectation_by_exact_enumeration():
    n, k = 4, 3
    _, _, q, b, a, _ = sampling_coefficients(n, k)
    p = np.array([.1, .2, .3, .4])
    reward = lambda draws: .4 * bool(set(draws) & {0, 1}) + .6 * bool(set(draws) & {3})
    expectation = np.zeros(n)
    gradient = np.zeros(n)
    for m in range(1, k + 1):
        for draws in itertools.product(range(n), repeat=m):
            probability = q[m - 1] * np.prod(p[list(draws)])
            counts = np.bincount(draws, minlength=n)
            expectation += probability * shifted_feedback(
                counts, p, m, reward(draws), q, a)
        for prefix in itertools.product(range(n), repeat=m - 1):
            probability = np.prod(p[list(prefix)])
            for element in range(n):
                gradient[element] += a[m - 1] * m * probability * reward(prefix + (element,))
    np.testing.assert_allclose(expectation, gradient - b.sum(), atol=1e-12)


def test_exact_log_domain_update_and_sampling():
    policy = SubEW(7, 3, 5000, 93)
    expected_eta = math.sqrt(math.log(7 / 3) / (policy.T * policy.Vq))
    assert policy.eta == pytest.approx(expected_eta)
    element_counts = np.zeros(7)
    for _ in range(policy.T):
        action = policy.act()
        assert 1 <= len(action) <= policy.sampled_m <= 3
        assert policy.counts.sum() == policy.sampled_m
        element_counts += policy.counts
        policy.observe(1.0)
    np.testing.assert_allclose(policy.histogram / policy.T, policy.q, atol=.02)
    np.testing.assert_allclose(
        element_counts / element_counts.sum(), np.full(7, 1 / 7), atol=.02)
    assert np.isclose(policy.p.sum(), 1)


def test_protocol_guards():
    policy = SubEW(5, 2, 1, 4)
    with pytest.raises(RuntimeError):
        policy.observe(.5)
    policy.act()
    with pytest.raises(RuntimeError):
        policy.act()
    policy.observe(.5)
    with pytest.raises(RuntimeError):
        policy.act()
