import math

import numpy as np
import pytest

from algorithms.related_work import BanditIG, OpaqueGreedy, softmax


def test_opaque_greedy_uses_report_exploration_probability():
    policy = OpaqueGreedy(34, 2, 100_000, 1)
    expected = min(1.0, 2 * (34 * math.log(34) / 100_000) ** (1 / 3))
    assert np.isclose(policy.gamma, expected)
    assert policy.default_eta == pytest.approx(
        math.sqrt(2 * 2 * 34 * math.log(34) / (expected * 100_000)))


def test_opaque_greedy_tunes_rate_and_keeps_theoretical_exploration():
    policy = OpaqueGreedy(34, 2, 100_000, 1, eta_multiplier=4)
    theoretical = min(1.0, 2 * (34 * math.log(34) / 100_000) ** (1 / 3))
    assert policy.gamma == pytest.approx(theoretical)
    assert policy.eta == pytest.approx(4 * policy.default_eta)


def test_opaque_greedy_feeds_unscaled_observed_reward():
    policy = OpaqueGreedy(7, 2, 1000, 1)
    policy.explore, policy.stage, policy.arm = True, 1, 3
    policy.observe(0.4)
    assert np.isclose(policy.logw[1, 3], 0.0)
    assert np.allclose(np.delete(policy.logw[1], 3), -policy.eta * 0.4)


def test_banditig_literal_blackwell_vector_equals_scalar_shortcut():
    policy = BanditIG(6, 2, 1000, 2)
    theta = np.array([.05, .10, .15, .20, .25, .25])
    policy.logw[0] = np.log(theta)
    policy.stage, policy.arm = 0, 4
    policy.theta = theta.copy()
    before = policy.logw[0].copy()
    reward = 0.7
    policy.observe(reward)

    scalar = before.copy()
    scalar[4] += math.sqrt(2 * math.log(policy.n)) * reward
    scalar -= scalar.max()
    assert np.allclose(policy.logw[0], scalar)
    expected = policy.n * reward * (theta[4] * np.ones(policy.n) - np.eye(policy.n)[4])
    assert np.allclose(policy.last_payoff, expected)


def test_banditig_uses_stage_local_update_clock_and_rate_multiplier():
    policy = BanditIG(6, 2, 1000, 2, eta_multiplier=4)
    policy.stage, policy.arm = 1, 3
    policy.theta = np.full(6, 1 / 6)
    policy.updates[0] = 100
    policy.observe(.5)
    assert policy.updates.tolist() == [100, 1]
    expected_gap = 4 * math.sqrt(2 * math.log(6)) * .5
    assert policy.logw[1, 3] - policy.logw[1, 0] == pytest.approx(expected_gap)


def test_banditig_uses_theorem_three_exploration_probability():
    policy = BanditIG(34, 2, 100_000, 2)
    expected = min(1.0, (34 ** 2 * math.log(34) / 100_000) ** (1 / 3))
    assert np.isclose(policy.q, expected)


def test_banditig_tunes_rate_and_keeps_theoretical_exploration():
    policy = BanditIG(34, 2, 100_000, 2, eta_multiplier=4)
    theoretical = min(1.0, (34 ** 2 * math.log(34) / 100_000) ** (1 / 3))
    assert policy.q == pytest.approx(theoretical)
    assert policy.eta_multiplier == 4


def test_banditig_nonexploration_does_not_advance_blackwell_state():
    policy = BanditIG(5, 2, 100, 3)
    before = policy.logw.copy()
    policy.stage = None
    policy.observe(0.9)
    assert np.array_equal(policy.logw, before)
    assert policy.t == 1


def test_banditig_distribution_is_reused_without_feedback():
    policy = BanditIG(5, 2, 100, 4)
    theta = softmax(policy.logw[0:1])[0]
    policy.stage = None
    policy.observe(0.5)
    assert np.allclose(softmax(policy.logw[0:1])[0], theta)
