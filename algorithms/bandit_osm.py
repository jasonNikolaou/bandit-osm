"""Exponential-weights algorithm from the Bandit OSM paper."""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import minimize


def coefficients_for_q(n: int, k: int, q):
    """Compute the paper's potential coefficients and exact variance bound."""
    q = np.asarray(q, dtype=float)
    if (q.shape != (k,) or not np.all(np.isfinite(q)) or np.any(q < 0)
            or abs(q.sum() - 1) > 1e-10):
        raise ValueError("q must be a probability distribution over draw counts")
    rho = 1 - 1 / k
    m = np.arange(1, k + 1, dtype=float)
    b = np.empty(k)
    tail = 0.0
    for j in range(k - 1, -1, -1):
        tail = q[j] + rho * tail
        b[j] = tail
    a = b / m
    numerator = b * b * (n + m - 1) / m
    terms = np.divide(numerator, q, out=np.full(k, np.inf), where=q > 0)
    terms[(q == 0) & (numerator == 0)] = 0
    variance = float(np.sum(terms))
    return b, a, variance


def minimum_variance_coefficients(n: int, k: int):
    """Minimize exact V_q subject to alpha(q) >= 1-1/e and full support."""
    rho, beta, initial, _, _, _ = sampling_coefficients(n, k)
    alpha_weights = 1 - rho ** np.arange(1, k + 1)
    target = 1 - 1 / math.e
    m = np.arange(1, k + 1, dtype=float)
    weights = (n + m - 1) / m / n
    transition = np.triu(rho ** np.subtract.outer(m, m).T)

    def objective(q):
        b = transition @ q
        terms = weights * b / q
        return float(np.dot(terms, b)), 2 * transition.T @ terms - terms * b / q

    # The quadratic-over-linear objective is convex on the positive simplex.
    result = minimize(
        objective, initial, jac=True,
        method="SLSQP", bounds=[(1e-10, 1)] * k,
        constraints=[
            {"type": "eq", "fun": lambda q: np.sum(q) - 1,
             "jac": lambda q: np.ones(k)},
            {"type": "ineq", "fun": lambda q: np.dot(alpha_weights, q) - target,
             "jac": lambda q: alpha_weights},
        ],
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    q = result.x
    if (not result.success or np.min(q) <= 0 or
            abs(np.sum(q) - 1) > 1e-8 or
            np.dot(alpha_weights, q) < target - 1e-9):
        raise RuntimeError(f"Constrained q optimization failed: {result.message}")
    b, a, variance = coefficients_for_q(n, k, q)
    return rho, beta, q, b, a, variance


def sampling_coefficients(n: int, k: int):
    """Return rho, beta, q, b, a, and the exact variance bound V_q."""
    if not 2 <= k <= n:
        raise ValueError("Require 2 <= k <= n")
    rho = 1 - 1 / k
    beta = 1 / math.e - rho ** k
    m = np.arange(1, k + 1, dtype=float)
    q = np.empty(k)
    q[:-1] = beta / (np.sum(1 / np.sqrt(m[:-1])) * np.sqrt(m[:-1]))
    q[-1] = 1 - beta
    b, a, variance = coefficients_for_q(n, k, q)
    return rho, beta, q, b, a, variance


def shifted_feedback(counts, probabilities, sample_size, reward, q, a):
    """Paper estimator with reward shifted from [0,1] to [-1,0]."""
    result = np.zeros_like(probabilities)
    support = counts > 0
    result[support] = (
        a[sample_size - 1] / q[sample_size - 1]
        * (reward - 1) * counts[support] / probabilities[support]
    )
    return result


class BanditOSM:
    """Vanilla log-domain EW with IID sampling and scalar feedback."""

    def __init__(self, n, k, horizon, seed, *, eta=None, eta_multiplier=1.0,
                 q_mode="classic", q=None):
        self.n, self.k, self.T = int(n), int(k), int(horizon)
        self.rng = np.random.default_rng(seed)
        if q_mode == "classic":
            coefficients = sampling_coefficients(self.n, self.k)
        elif q_mode == "min_variance":
            coefficients = minimum_variance_coefficients(self.n, self.k)
        elif q_mode == "custom":
            if q is None:
                raise ValueError("Custom q mode requires q")
            rho = 1 - 1 / self.k
            beta = 1 / math.e - rho ** self.k
            distribution = np.asarray(q, dtype=float)
            b, a, variance = coefficients_for_q(self.n, self.k, distribution)
            coefficients = rho, beta, distribution, b, a, variance
        else:
            raise ValueError(f"Unknown q mode: {q_mode}")
        self.rho, self.beta, self.q, self.b, self.a, self.Vq = coefficients
        self.default_eta = (math.sqrt(
            math.log(self.n / self.k) / (self.T * self.Vq))
            if math.isfinite(self.Vq) else None)
        self.eta_multiplier = float(eta_multiplier)
        if eta is None and self.default_eta is None:
            raise ValueError("Explicit eta required when V_q is infinite")
        self.eta = float(eta if eta is not None
                         else self.eta_multiplier * self.default_eta)
        if not math.isfinite(self.eta) or self.eta <= 0:
            raise ValueError("eta must be finite and positive")
        self.logp = np.full(self.n, -math.log(self.n))
        self.p = np.exp(self.logp)
        self.t = 0
        self.pending = False
        self.histogram = np.zeros(self.k, dtype=int)
        self.min_log_probability = float(self.logp.min())
        self.underflow_rounds = 0
        self.parameters = {
            "source": "Bandit_OSM (7).pdf Algorithm 2",
            "eta": self.eta, "default_eta": self.default_eta,
            "eta_multiplier": self.eta_multiplier,
            "rho": self.rho, "beta": self.beta,
            "q_mode": q_mode,
            "alpha_q": float(np.dot(
                self.q, 1 - self.rho ** np.arange(1, self.k + 1))),
            "q": self.q.tolist(), "b": self.b.tolist(),
            "a": self.a.tolist(),
            "V_q": self.Vq if math.isfinite(self.Vq) else None,
            "sampling": "M~q; M IID draws with replacement; play their union",
            "reward_shift": -1,
        }

    def act(self):
        if self.pending or self.t >= self.T:
            raise RuntimeError("Invalid act/observe sequence")
        self.sampled_m = 1 + int(self.rng.choice(self.k, p=self.q))
        draws = self.rng.choice(self.n, size=self.sampled_m, replace=True, p=self.p)
        self.counts = np.bincount(draws, minlength=self.n)
        self.histogram[self.sampled_m - 1] += 1
        self.pending = True
        return np.flatnonzero(self.counts)

    def observe(self, reward):
        if not self.pending:
            raise RuntimeError("Call act before observe")
        feedback = shifted_feedback(
            self.counts, self.p, self.sampled_m, float(reward), self.q, self.a)
        logits = self.logp + self.eta * feedback
        logits -= logits.max()
        self.logp = logits - math.log(float(np.exp(logits).sum()))
        with np.errstate(under="ignore"):
            self.p = np.exp(self.logp)
        self.min_log_probability = min(
            self.min_log_probability, float(self.logp.min()))
        self.underflow_rounds += int(np.any(self.p == 0))
        self.pending = False
        self.t += 1

    def iterate(self):
        return self.logp

    def diagnostics(self):
        return {
            "draw_count_histogram": self.histogram.tolist(),
            "minimum_log_probability": self.min_log_probability,
            "sampling_cache_underflow_rounds": self.underflow_rounds,
            "final_probability_sum": float(self.p.sum()),
        }
