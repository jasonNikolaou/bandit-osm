"""Cardinality specializations of the two Wan adversarial bandit algorithms.

Wan et al. (ICML 2023), Algorithms 2 and 4, use one partition group of
capacity k. Wan and Zhang (2026), Algorithm 2, uses the uniform rank-k matroid.
Both implementations query only the reward of the feasible set they play.
"""
from __future__ import annotations

import math

import numpy as np


ALPHA = 1 - 1 / math.e


def _barrier_direction(probabilities, rng):
    """Sample H v for H=(nabla^2 Phi)^(-1/2), v uniform on S^(d-1).

    Each simplex barrier has Hessian diag(1/x_i^2)+11'/slack^2.
    Projecting an (n+1)-variate Gaussian orthogonally to (x,slack), then
    scaling its first n coordinates by x, gives covariance Hessian^-1.
    Normalizing the projected Gaussian across all k blocks gives H v in
    distribution without forming a dense matrix square root.
    """
    k, n = probabilities.shape
    slack = 1 - probabilities.sum(axis=1)
    augmented = np.column_stack((probabilities, slack))
    gaussian = rng.normal(size=(k, n + 1))
    projection = np.sum(augmented * gaussian, axis=1)
    projection /= np.sum(augmented * augmented, axis=1)
    gaussian -= augmented * projection[:, None]
    gaussian /= np.linalg.norm(gaussian)
    return probabilities * gaussian[:, :n]


def _barrier_hessian_times(probabilities, direction):
    slack = 1 - probabilities.sum(axis=1)
    return (direction / probabilities**2
            + direction.sum(axis=1)[:, None] / slack[:, None]**2)


def _simplex_barrier_ftrl(cumulative_gradient, eta):
    """Exact scalar-root solution of min_x Phi(x)-eta<g,x> per simplex."""
    result = np.empty_like(cumulative_gradient)
    for stage, gradient in enumerate(cumulative_gradient):
        a = eta * gradient
        maximum = float(a.max())
        low, high = 0.0, len(a) + 1.0
        if maximum >= 0:
            # Solve for gap=lambda-max(a). This avoids cancellation when
            # the accumulated linear term is large and positive.
            offsets = maximum - a
            for _ in range(70):
                middle = (low + high) / 2
                if (1 / (maximum + middle)
                        + np.sum(1 / (middle + offsets))) > 1:
                    low = middle
                else:
                    high = middle
            gap = (low + high) / 2
            result[stage] = 1 / (gap + offsets)
        else:
            for _ in range(70):
                middle = (low + high) / 2
                if 1 / middle + np.sum(1 / (middle - a)) > 1:
                    low = middle
                else:
                    high = middle
            lam = (low + high) / 2
            result[stage] = 1 / (lam - a)
        if 1 - result[stage].sum() <= 1e-14:
            raise FloatingPointError(
                "Barrier iterate is too close to a simplex boundary")
    return result


class BanditMLSM:
    """Wan et al. (2023), MLSMWrapper with BanditMLSM4PS, one group.

    Lemma 5.1 uses eta=d^(-7/3) T^(-1/3) and L=d^(-5/3) T^(1/3)
    for this wrapper. Its block length is below one at our dimensions.
    We tune L and use Algorithm 1's eta as a practical numerical base;
    this class does not claim the wrapper theorem's finite-T bound.
    """

    def __init__(self, n, k, horizon, seed, *, eta_multiplier=1.0,
                 block_size=32, base_eta=None):
        self.n, self.k, self.T = int(n), int(k), int(horizon)
        self.d = self.n * self.k
        self.rng = np.random.default_rng(seed)
        self.block_size = int(block_size)
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        # Lemma 5.1's wrapper formulas are recorded for attribution only.
        self.wrapper_theoretical_eta = self.d**(-7 / 3) * self.T**(-1 / 3)
        self.wrapper_theoretical_block_size = (
            self.d**(-5 / 3) * self.T**(1 / 3))
        # Algorithm 1 / Theorem 3.3 supplies a practical tuning reference,
        # not the prescribed learning rate for the product-simplex wrapper.
        self.algorithm1_eta_reference = self.d**-4 * self.T**(-2 / 3)
        self.base_eta = (float(base_eta) if base_eta is not None
                         else self.algorithm1_eta_reference)
        self.eta_multiplier = float(eta_multiplier)
        self.eta = self.base_eta * self.eta_multiplier
        if not math.isfinite(self.eta) or self.eta <= 0:
            raise ValueError("eta must be finite and positive")
        self.x = np.full((self.k, self.n), 1 / (self.n + 1))
        self.next_x = None
        self.cumulative_gradient = np.zeros((self.k, self.n))
        self.t = 0
        self.pending = False
        self.exploration_rounds = 0
        self._choose_exploration_round()
        self.parameters = {
            "source": "Wan et al. (2023), Algorithms 2 and 4; one partition group",
            "block_size": self.block_size,
            "eta": self.eta,
            "eta_base": self.base_eta,
            "eta_base_source": (
                "explicit override" if base_eta is not None else
                "practical reference from Algorithm 1 / Theorem 3.3"),
            "wrapper_theoretical_eta": self.wrapper_theoretical_eta,
            "wrapper_theoretical_block_size": self.wrapper_theoretical_block_size,
            "wrapper_parameter_source": "Wan et al. (2023), proof of Lemma 5.1",
            "eta_multiplier": self.eta_multiplier,
            "extension": "k independent standard simplexes with a dummy element",
            "iterate": "sum of k simplex-coordinate vectors",
            "block_update": "activate the new iterate at the next block boundary",
        }

    def _choose_exploration_round(self):
        remaining = min(self.block_size, self.T - self.t)
        self.exploration_offset = (int(self.rng.integers(remaining))
                                   if remaining else -1)

    def _sample_extension(self, point):
        draws = []
        for simplex in point:
            probabilities = np.append(simplex, max(0.0, 1 - simplex.sum()))
            probabilities /= probabilities.sum()
            item = int(self.rng.choice(self.n + 1, p=probabilities))
            if item != self.n:
                draws.append(item)
        return np.unique(draws)

    def act(self):
        if self.pending or self.t >= self.T:
            raise RuntimeError("Invalid act/observe sequence")
        self.exploring = self.t % self.block_size == self.exploration_offset
        point = self.x
        if self.exploring:
            self.exploration_rounds += 1
            z = math.log1p(self.rng.random() * (math.e - 1))
            direction = _barrier_direction(self.x, self.rng)
            coordinate = int(self.rng.integers(self.d))
            stage, item = divmod(coordinate, self.n)
            if z >= .5:
                # Algorithm 4 uses probability 1/2 for u=0 and 1/(2d)
                # for each e_i: sample the branch first, then a coordinate.
                positive = self.rng.random() < .5
                if positive:
                    point = z * self.x.copy()
                    point[stage, item] += z * direction[stage, item]
                    self.scalar_factor = 2 * ALPHA * self.d / z
                else:
                    point = z * self.x
                    self.scalar_factor = -2 * ALPHA * self.d / z
            else:
                positive = self.rng.random() < .5
                point = z * self.x.copy()
                if positive:
                    point[stage, item] += .5
                self.scalar_factor = ((1 if positive else -1)
                                      * 4 * ALPHA * self.d
                                      * direction[stage, item])
            self.inverse_direction = _barrier_hessian_times(
                self.x, direction)
        self.pending = True
        return self._sample_extension(point)

    def observe(self, reward):
        if not self.pending:
            raise RuntimeError("Call act before observe")
        if self.exploring:
            estimate = self.d * self.scalar_factor * float(reward)
            self.cumulative_gradient += estimate * self.inverse_direction
            self.next_x = _simplex_barrier_ftrl(
                self.cumulative_gradient, self.eta)
        self.t += 1
        self.pending = False
        if self.t < self.T and self.t % self.block_size == 0:
            if self.next_x is None:
                raise RuntimeError("Block ended without an exploration update")
            self.x, self.next_x = self.next_x, None
            self._choose_exploration_round()

    def iterate(self):
        return self.x.sum(axis=0)

    def diagnostics(self):
        return {
            "exploration_rounds": self.exploration_rounds,
            "minimum_simplex_coordinate": float(self.x.min()),
            "minimum_dummy_probability": float((1 - self.x.sum(axis=1)).min()),
        }


def _capped_entropy_projection(log_weights, total, lower, upper):
    """KL projection onto {sum x=total, lower<=x_i<=upper}."""
    log_lower, log_upper = math.log(lower), math.log(upper)
    low = float(np.min(log_lower - log_weights))
    high = float(np.max(log_upper - log_weights))
    for _ in range(65):
        middle = (low + high) / 2
        values = np.exp(np.clip(log_weights + middle, log_lower, log_upper))
        if values.sum() < total:
            low = middle
        else:
            high = middle
    return np.exp(np.clip(log_weights + (low + high) / 2,
                          log_lower, log_upper))


class PoissonBE:
    """Wan and Zhang (2026), Algorithm 2 for a uniform rank-k matroid."""

    def __init__(self, n, k, horizon, seed, *, eta_multiplier=1.0,
                 exploration_probability=.25, shrinkage=None,
                 starting_time=None, event_cap=None):
        self.n, self.k, self.T = int(n), int(k), int(horizon)
        self.rng = np.random.default_rng(seed)
        self.epsilon = (1 / self.T if starting_time is None
                        else float(starting_time))
        self.mu = self.k * math.log(1 / self.epsilon)
        self.gamma = float(exploration_probability)
        self.delta = (min(.5, 1 / (self.mu * self.T)) if shrinkage is None
                      else float(shrinkage))
        self.cap = (math.ceil(2 * math.e * (self.mu + math.log(self.T)))
                    if event_cap is None else int(event_cap))
        if not (0 < self.epsilon < 1 and 0 < self.gamma <= 1
                and 0 < self.delta < 1 and self.cap >= 0):
            raise ValueError("Invalid Poisson-bandit parameters")
        self.paper_gamma = ((self.n * (self.mu**2 + self.mu)
                             * math.log(math.e * self.n) / self.T)**(1 / 3))
        self.default_eta = math.sqrt(
            self.k**2 * self.gamma * math.log(math.e * self.n)
            / (2 * self.n * (self.mu**2 + self.mu) * self.T))
        self.eta_multiplier = float(eta_multiplier)
        self.eta = self.default_eta * self.eta_multiplier
        if not math.isfinite(self.eta) or self.eta <= 0:
            raise ValueError("eta must be finite and positive")
        self.x = np.full(self.n, self.k / self.n)
        self.lower = self.delta * self.k / self.n
        self.upper = 1 - self.delta + self.lower
        self.start_base = np.sort(self.rng.choice(self.n, self.k, replace=False))
        self.t = 0
        self.pending = False
        self.exploration_rounds = 0
        self.overflow_rounds = 0
        self.zero_event_rounds = 0
        self.parameters = {
            "source": "Wan and Zhang (2026), Algorithm 2; uniform matroid",
            "epsilon": self.epsilon,
            "mu": self.mu,
            "gamma": self.gamma,
            "paper_gamma": self.paper_gamma,
            "delta": self.delta,
            "eta": self.eta,
            "default_eta": self.default_eta,
            "eta_multiplier": self.eta_multiplier,
            "event_cap": self.cap,
            "balanced_exchange": "uniform-matroid closed-form coupling",
            "projection": "exact scalar-root capped-simplex entropy projection",
        }

    def _exchange(self, base, cdf):
        # Column sums of Q are x_j, so first draw j from x/k. If j is in
        # A the only feasible deletion is i=j. Otherwise row residuals
        # are 1-x_i, and Q(i,j)=(1-x_i)x_j/sum_{h notin A}x_h.
        j = int(np.searchsorted(cdf, self.rng.random(), side="right"))
        if j == self.n:
            j -= 1
        position = np.flatnonzero(base == j)
        if len(position):
            return int(position[0]), j
        deletion_weights = 1 - self.x[base]
        deletion_weights /= deletion_weights.sum()
        return int(self.rng.choice(self.k, p=deletion_weights)), j

    def act(self):
        if self.pending or self.t >= self.T:
            raise RuntimeError("Invalid act/observe sequence")
        m = int(self.rng.poisson(self.mu))
        self.feedback_data = None
        if m > self.cap:
            self.overflow_rounds += 1
            action = self.start_base.copy()
        elif m == 0:
            self.zero_event_rounds += 1
            action = self.start_base.copy()
        else:
            times = np.sort(self.epsilon**(1 - self.rng.random(m)))
            base = self.start_base.copy()
            cdf = np.cumsum(self.x / self.k)
            cdf[-1] = 1
            explore = self.rng.random() < self.gamma
            chosen = int(self.rng.integers(m)) if explore else -1
            for event, tau in enumerate(times):
                position, j = self._exchange(base, cdf)
                if event == chosen:
                    prior = base.copy()
                    removed = int(prior[position])
                    self.feedback_data = (m, float(tau), j)
                base[position] = j
            if explore:
                self.exploration_rounds += 1
                retained = prior[prior != removed]
                retained = retained[self.rng.random(len(retained)) < times[chosen]]
                self.plus_branch = self.rng.random() < .5
                action = (np.append(retained, self.feedback_data[2])
                          if self.plus_branch else retained)
            else:
                action = base
        self.pending = True
        return np.sort(np.asarray(action, dtype=int))

    def observe(self, reward):
        if not self.pending:
            raise RuntimeError("Call act before observe")
        if self.feedback_data is not None:
            m, tau, j = self.feedback_data
            z = (2 * (1 - float(reward)) if self.plus_branch
                 else 2 * float(reward))
            loss = m * tau * math.exp(-(1 - tau)) * z / (self.gamma * self.x[j])
            log_weights = np.log(self.x)
            log_weights[j] -= self.eta * loss
            self.x = _capped_entropy_projection(
                log_weights, self.k, self.lower, self.upper)
        self.pending = False
        self.t += 1

    def iterate(self):
        return self.x

    def diagnostics(self):
        return {
            "exploration_rounds": self.exploration_rounds,
            "overflow_rounds": self.overflow_rounds,
            "zero_event_rounds": self.zero_event_rounds,
            "minimum_fractional_coordinate": float(self.x.min()),
            "fractional_base_sum": float(self.x.sum()),
        }
