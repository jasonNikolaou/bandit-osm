"""Retained adversarial scalar-bandit baselines."""
from __future__ import annotations

import math
import numpy as np

from .sub_ew import SubEW


def softmax(log_weights):
    weights = np.exp(log_weights - np.max(log_weights, axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


class Policy:
    def __init__(self, n, k, horizon, seed, *, eta_multiplier=1.0):
        self.n, self.k, self.T = int(n), int(k), int(horizon)
        self.rng = np.random.default_rng(seed)
        self.eta_multiplier = float(eta_multiplier)
        self.t = 0
        self.parameters = {}

    def iterate(self):
        return np.empty(0)

    def diagnostics(self):
        return {}


class RandomPolicy(Policy):
    def act(self):
        return self.rng.choice(self.n, self.k, replace=False)

    def observe(self, reward):
        self.t += 1


class MSE3(Policy):
    """Pasteris et al. (2023), Algorithm 1, with zero costs."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logw = np.zeros(self.n)
        self.default_eta = math.sqrt(
            math.log(self.n) / (2 * self.k * (self.n + self.k - 1) * self.T))
        self.eta = self.eta_multiplier * self.default_eta
        self.parameters = {
            "eta": self.eta, "default_eta": self.default_eta,
            "eta_multiplier": self.eta_multiplier,
            "reward_shift": -1, "costs": 0,
        }

    def act(self):
        self.p = softmax(self.logw)
        draws = self.rng.choice(self.n, self.k, p=self.p)
        self.counts = np.bincount(draws, minlength=self.n)
        return np.flatnonzero(self.counts)

    def observe(self, reward):
        support = self.counts > 0
        self.logw[support] += (
            self.eta * (reward - 1) * self.counts[support] / self.p[support])
        self.logw -= self.logw.max()
        self.t += 1

    def iterate(self):
        return self.logw


class OpaqueGreedy(Policy):
    """Streeter-Golovin opaque-feedback greedy specialization."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logw = np.zeros((self.k, self.n))
        self.default_gamma = min(
            1.0, self.k * (self.n * math.log(self.n) / self.T) ** (1 / 3))
        self.gamma = self.default_gamma
        # WMR receives a nonzero payoff only when its stage and arm are the
        # uniformly selected opaque-feedback probe.  This is the standard
        # fixed-rate form of WMR for that sparse payoff sequence.
        self.default_eta = math.sqrt(
            2 * self.k * self.n * math.log(self.n) / (self.gamma * self.T))
        self.eta = self.eta_multiplier * self.default_eta
        self.parameters = {
            "gamma": self.gamma, "default_gamma": self.default_gamma,
            "eta": self.eta, "default_eta": self.default_eta,
            "eta_multiplier": self.eta_multiplier,
            "source": "CMU-CS-07-171 Section 4.4.3",
            "feedback": "unscaled observed reward to the probed stage/arm",
        }

    def act(self):
        distributions = softmax(self.logw)
        self.chosen = [self.rng.choice(self.n, p=p) for p in distributions]
        self.explore = self.rng.random() < self.gamma
        if not self.explore:
            return np.unique(self.chosen)
        self.stage = int(self.rng.integers(self.k))
        self.arm = int(self.rng.integers(self.n))
        return np.unique(self.chosen[:self.stage] + [self.arm])

    def observe(self, reward):
        if self.explore:
            self.logw[self.stage, self.arm] += self.eta * reward
            self.logw[self.stage] -= self.logw[self.stage].max()
        self.t += 1

    def iterate(self):
        return self.logw


class BanditIG(Policy):
    """Niazadeh et al. (2023), Algorithms 3-4 and Example 1."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logw = np.zeros((self.k, self.n))
        self.default_q = min(
            1.0, (self.n ** 2 * math.log(self.n) / self.T) ** (1 / 3))
        self.q = self.default_q
        self.updates = np.zeros(self.k, dtype=int)
        self.parameters = {
            "q": self.q, "default_q": self.default_q,
            "eta_multiplier": self.eta_multiplier,
            "eta_schedule": (
                "eta_multiplier*q/n*sqrt(2*log(n)/stage_update_count); "
                "applied to p_hat/q"),
            "source": "Niazadeh et al. Algorithms 3-4, Example 1",
        }

    def act(self):
        prefix = []
        self.stage = None
        for stage, distribution in enumerate(softmax(self.logw)):
            if self.rng.random() < self.q:
                self.stage = stage
                self.arm = int(self.rng.integers(self.n))
                self.theta = distribution.copy()
                return np.unique(prefix + [self.arm])
            prefix.append(int(self.rng.choice(self.n, p=distribution)))
        return np.unique(prefix)

    def observe(self, reward):
        if self.stage is not None:
            payoff = self.n * reward * self.theta[self.arm] * np.ones(self.n)
            payoff[self.arm] -= self.n * reward
            self.last_payoff = payoff
            self.updates[self.stage] += 1
            eta = (self.eta_multiplier * self.q / self.n
                   * math.sqrt(2 * math.log(self.n) / self.updates[self.stage]))
            self.logw[self.stage] -= eta * payoff / self.q
            self.logw[self.stage] -= self.logw[self.stage].max()
        self.t += 1

    def iterate(self):
        return self.logw

    def diagnostics(self):
        return {"updates_by_stage": self.updates.tolist()}


POLICIES = {
    "Sub-EW": SubEW,
    "Random": RandomPolicy,
    "MSE3": MSE3,
    "OpaqueGreedy": OpaqueGreedy,
    "Bandit-IG": BanditIG,
}
