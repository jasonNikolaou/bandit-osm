"""Canonical Influence, Team Formation, and Weighted Coverage instances."""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from algorithms.sub_ew import sampling_coefficients


def _knn(embeddings, neighbors):
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = normalized @ normalized.T
    np.fill_diagonal(similarities, -np.inf)
    edges = set()
    for i in range(len(embeddings)):
        for j in np.argpartition(-similarities[i], neighbors - 1)[:neighbors]:
            edges.add((min(i, int(j)), max(i, int(j))))
    np.fill_diagonal(similarities, 1)
    return np.asarray(sorted(edges), dtype=np.int32), similarities


def teamformation(*, horizon, seed, n=100, k=3, task_types=10, top_q=10,
                   dimension=8, neighbors=15, max_value=100.0,
                   noise=10.0, score_scale=40.0, low_fraction=.5,
                   tail_low=.0005, tail_high=.005,
                   overlap_low=.05, overlap_high=.15,
                   requirement_spread=None):
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(n, dimension))
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    edges, similarities = _knn(embeddings, neighbors)
    low_count = round(low_fraction * n)
    low_elements = np.sort(rng.choice(n, low_count, replace=False)).astype(np.int32)
    low_mask = np.zeros(n, dtype=bool)
    low_mask[low_elements] = True
    eligible = np.flatnonzero(~low_mask)
    requirements = np.empty((task_types, dimension))
    common_requirement = None
    if requirement_spread is not None:
        common_requirement = rng.normal(size=dimension)
        common_requirement /= np.linalg.norm(common_requirement)
    weights = np.empty((task_types, n))
    penalties = np.zeros((task_types, n, n))
    top_candidates = np.empty((task_types, top_q), dtype=np.int32)
    overlap = rng.uniform(overlap_low, overlap_high, size=task_types)
    for task in range(task_types):
        task_direction = rng.normal(size=dimension)
        task_direction /= np.linalg.norm(task_direction)
        requirement = task_direction
        if common_requirement is not None:
            requirement = common_requirement + requirement_spread * task_direction
            requirement /= np.linalg.norm(requirement)
        requirements[task] = requirement
        scores = score_scale * (normalized @ requirement)
        scores += rng.normal(scale=noise, size=n)
        scores -= scores.min()
        scores *= max_value / scores.max()
        eligible_scores = scores[eligible]
        selected = eligible[np.argpartition(eligible_scores, -top_q)[-top_q:]]
        selected = selected[np.argsort(-scores[selected], kind="stable")]
        top_candidates[task] = selected
        weights[task] = max_value * rng.uniform(tail_low, tail_high, size=n)
        weights[task, selected] = scores[selected]
        for i, j in edges:
            similarity = max(0.0, similarities[i, j])
            penalties[task, i, j] = penalties[task, j, i] = (
                -overlap[task] * similarity * min(weights[task, i], weights[task, j])
                / (k - 1))
    task_ids = rng.choice(task_types, horizon).astype(np.int16)
    normalization = float(np.sort(weights, axis=1)[:, -k:].sum(axis=1).max())
    return {
        "kind": "teamformation", "n": n, "k": k, "horizon": horizon,
        "normalization": normalization, "embeddings": embeddings,
        "requirements": requirements, "edges": edges, "weights": weights,
        "penalties": penalties, "top_candidates": top_candidates,
        "low_elements": low_elements, "low_mask": low_mask,
        "overlap_strengths": overlap, "task_ids": task_ids,
        "common_requirement": (
            np.zeros(dimension) if common_requirement is None
            else common_requirement),
        "requirement_spread": np.asarray(
            -1.0 if requirement_spread is None else requirement_spread),
    }


def coverage(*, horizon, seed, n=100, k=3, task_types=10,
                      topics=10, features_per_topic=20,
                      background_features=50, low_fraction=.75,
                      shared_topics=0, specific_low=.15,
                      specific_high=.30, specialists_per_topic=1,
                      shared_generalists=0, shared_features_per_topic=8):
    rng = np.random.default_rng(seed)
    topic_count = topics * features_per_topic
    feature_count = topic_count + background_features
    low_count = round(low_fraction * n)
    permutation = rng.permutation(n)
    low_elements = np.sort(permutation[:low_count]).astype(np.int32)
    specialist_count = topics * specialists_per_topic
    specialists = permutation[low_count:low_count + specialist_count]
    generalists = permutation[low_count + specialist_count:]
    low_mask = np.zeros(n, dtype=bool)
    low_mask[low_elements] = True
    coverage = np.zeros((n, feature_count), dtype=bool)
    background = np.arange(topic_count, feature_count)
    specialist_topics = np.repeat(
        np.arange(topics, dtype=np.int16), specialists_per_topic)
    for topic, element in zip(specialist_topics, specialists):
        pool = np.arange(topic * features_per_topic, (topic + 1) * features_per_topic)
        coverage[element, rng.choice(pool, min(16, len(pool)), replace=False)] = True
        coverage[element, rng.choice(background, min(2, len(background)), replace=False)] = True
    for position, element in enumerate(generalists):
        if position < shared_generalists:
            for topic in range(shared_topics):
                pool = np.arange(
                    topic * features_per_topic,
                    (topic + 1) * features_per_topic)
                coverage[element, rng.choice(
                    pool, min(shared_features_per_topic, len(pool)),
                    replace=False)] = True
            coverage[element, rng.choice(
                background, min(2, len(background)), replace=False)] = True
            continue
        for topic in rng.choice(topics, min(3, topics), replace=False):
            pool = np.arange(topic * features_per_topic, (topic + 1) * features_per_topic)
            coverage[element, rng.choice(pool, min(3, len(pool)), replace=False)] = True
        coverage[element, rng.choice(background, min(2, len(background)), replace=False)] = True
    for element in low_elements:
        coverage[element, rng.choice(background, min(2, len(background)), replace=False)] = True
    feature_weights = np.empty((task_types, feature_count))
    specific_topics = np.full(task_types, -1, dtype=np.int16)
    for task in range(task_types):
        values = rng.uniform(.01, .03, feature_count)
        values[background] = rng.uniform(.001, .005, background_features)
        if shared_topics:
            for topic in range(shared_topics):
                shared = slice(
                    topic * features_per_topic, (topic + 1) * features_per_topic)
                values[shared] = rng.uniform(.8, 1.2, features_per_topic)
            specific_topic = shared_topics + task % (topics - shared_topics)
            specific_topics[task] = specific_topic
            specific = slice(
                specific_topic * features_per_topic,
                (specific_topic + 1) * features_per_topic)
            values[specific] = rng.uniform(
                specific_low, specific_high, features_per_topic)
        else:
            primary = slice(
                task * features_per_topic, (task + 1) * features_per_topic)
            values[primary] = rng.uniform(.8, 1.2, features_per_topic)
        feature_weights[task] = values
    task_ids = rng.choice(task_types, horizon).astype(np.int16)
    specialist_order = np.argsort(specialists)
    return {
        "kind": "coverage", "n": n, "k": k, "horizon": horizon,
        "normalization": float(feature_weights.sum(axis=1).max()),
        "coverage": coverage, "feature_weights": feature_weights,
        "low_elements": low_elements, "low_mask": low_mask,
        "specialists": specialists[specialist_order].astype(np.int32),
        "specialist_topics": specialist_topics[specialist_order],
        "generalists": np.sort(generalists).astype(np.int32),
        "shared_topics": np.asarray(shared_topics),
        "shared_generalist_elements": np.sort(
            generalists[:shared_generalists]).astype(np.int32),
        "specialists_per_topic": np.asarray(specialists_per_topic),
        "specific_topics": specific_topics,
        "task_ids": task_ids,
    }


def influence(*, horizon, seed, retention=.05, k=2):
    graph = nx.karate_club_graph()
    n = len(graph)
    edges = np.asarray(sorted(tuple(sorted(edge)) for edge in graph.edges()), dtype=np.int32)
    node_weights = np.full(n, .01)
    node_weights[[0, 33]] = .5
    total_weight = float(node_weights.sum())
    rng = np.random.default_rng(seed)
    labels = np.empty((horizon, n), dtype=np.int16)
    masses = np.empty((horizon, n), dtype=np.float64)
    sizes = np.empty((horizon, n), dtype=np.int16)
    for t in range(horizon):
        live = edges[rng.random(len(edges)) < retention]
        matrix = coo_matrix((np.ones(len(live)), (live[:, 0], live[:, 1])),
                            shape=(n, n)).tocsr()
        _, component = connected_components(matrix, directed=False)
        component_mass = np.bincount(component, weights=node_weights)
        component_size = np.bincount(component)
        labels[t] = component
        masses[t] = component_mass[component]
        sizes[t] = component_size[component]
    return {
        "kind": "influence", "n": n, "k": k, "horizon": horizon,
        "normalization": total_weight, "labels": labels, "masses": masses,
        "sizes": sizes,
        "node_weights": node_weights, "edges": edges,
        "retention": np.asarray(retention), "task_ids": np.arange(horizon),
    }


GENERATORS = {
    "influence": influence,
    "teamformation": teamformation,
    "coverage": coverage,
}


def resample_stream(instance, seed):
    """Return the fixed structural instance with a new stochastic stream."""
    kind = instance["kind"]
    horizon = int(instance["horizon"])
    if kind == "influence":
        return influence(
            horizon=horizon, seed=int(seed),
            retention=float(instance["retention"]), k=int(instance["k"]))
    if kind not in ("teamformation", "coverage"):
        raise ValueError(f"Unknown objective: {kind}")
    replica = dict(instance)
    task_count = len(
        instance["weights"] if kind == "teamformation"
        else instance["feature_weights"])
    replica["task_ids"] = np.random.default_rng(seed).choice(
        task_count, horizon).astype(np.int16)
    return replica


def reward(instance, t, action):
    action = np.unique(np.asarray(action, dtype=int))
    if len(action) == 0:
        return 0.0
    kind = instance["kind"]
    if kind == "teamformation":
        task = int(instance["task_ids"][t])
        matrix = instance["penalties"][task]
        raw = instance["weights"][task, action].sum()
        raw += matrix[np.ix_(action, action)].sum() / 2
    elif kind == "coverage":
        task = int(instance["task_ids"][t])
        covered = np.any(instance["coverage"][action], axis=0)
        raw = instance["feature_weights"][task, covered].sum()
    elif kind == "influence":
        chosen_labels = instance["labels"][t, action]
        _, first = np.unique(chosen_labels, return_index=True)
        raw = instance["masses"][t, action[first]].sum()
    else:
        raise ValueError(f"Unknown objective: {kind}")
    return float(raw / instance["normalization"])


def _candidate_sets(n, k):
    count = math.comb(n, k)
    values = np.fromiter(
        (element for subset in itertools.combinations(range(n), k)
         for element in subset),
        dtype=np.int32, count=count * k)
    return values.reshape(count, k)


def fixed_optimum(instance):
    """Exact best fixed cardinality-k set for the complete task stream."""
    kind = instance["kind"]
    horizon = int(instance["horizon"])
    n = int(instance["n"])
    k = int(instance["k"])
    candidates = _candidate_sets(n, k)
    if kind == "teamformation":
        counts = np.bincount(instance["task_ids"], minlength=len(instance["weights"]))
        frequency = counts / horizon
        mean_weights = frequency @ instance["weights"]
        mean_penalties = np.tensordot(frequency, instance["penalties"], axes=1)
        scores = mean_weights[candidates].sum(axis=1)
        for left in range(k):
            for right in range(left + 1, k):
                scores += mean_penalties[
                    candidates[:, left], candidates[:, right]]
    elif kind == "coverage":
        counts = np.bincount(
            instance["task_ids"], minlength=len(instance["feature_weights"]))
        mean_weights = counts @ instance["feature_weights"] / horizon
        scores = np.empty(len(candidates))
        for start in range(0, len(candidates), 10_000):
            stop = min(start + 10_000, len(candidates))
            covered = np.any(instance["coverage"][candidates[start:stop]], axis=1)
            scores[start:stop] = covered @ mean_weights
    elif kind == "influence":
        if k != 2:
            raise ValueError("The exact Influence optimizer currently requires k=2")
        left, right = candidates.T
        scores = np.empty(len(candidates))
        for start in range(0, len(candidates), 256):
            stop = min(start + 256, len(candidates))
            i, j = left[start:stop], right[start:stop]
            distinct = instance["labels"][:, i] != instance["labels"][:, j]
            values = instance["masses"][:, i]
            values += distinct * instance["masses"][:, j]
            scores[start:stop] = values.mean(axis=0)
    else:
        raise ValueError(f"Unknown objective: {kind}")
    chosen = candidates[int(np.argmax(scores))].tolist()
    trajectory = np.array([reward(instance, t, chosen) for t in range(horizon)])
    return chosen, trajectory


def references(instance):
    n, k = int(instance["n"]), int(instance["k"])
    kind = instance["kind"]
    q = sampling_coefficients(n, k)[2]
    if kind == "coverage":
        degrees = instance["coverage"].sum(axis=0)
        miss = np.ones(len(degrees))
        for draw in range(k):
            miss *= np.maximum(0, n - degrees - draw) / (n - draw)
        random_by_task = instance["feature_weights"] @ (1 - miss) / instance["normalization"]
        iid_cover = q @ np.array([
            1 - (1 - degrees / n) ** m for m in range(1, k + 1)])
        ew_by_task = instance["feature_weights"] @ iid_cover / instance["normalization"]
        ids = instance["task_ids"]
        random_values, ew_values = random_by_task[ids], ew_by_task[ids]
    elif kind == "teamformation":
        p1, p2 = k / n, k * (k - 1) / (n * (n - 1))
        singleton = q @ np.array([1 - (1 - 1 / n) ** m for m in range(1, k + 1)])
        pair = q @ np.array([
            1 - 2 * (1 - 1 / n) ** m + (1 - 2 / n) ** m
            for m in range(1, k + 1)])
        random_by_task = (p1 * instance["weights"].sum(axis=1)
            + p2 * np.triu(instance["penalties"], 1).sum(axis=(1, 2))) / instance["normalization"]
        ew_by_task = (singleton * instance["weights"].sum(axis=1)
            + pair * np.triu(instance["penalties"], 1).sum(axis=(1, 2))) / instance["normalization"]
        ids = instance["task_ids"]
        random_values, ew_values = random_by_task[ids], ew_by_task[ids]
    else:
        random_values = np.empty(instance["horizon"])
        ew_values = np.empty(instance["horizon"])
        for t in range(instance["horizon"]):
            _, first = np.unique(instance["labels"][t], return_index=True)
            sizes = instance["sizes"][t, first]
            masses = instance["masses"][t, first]
            miss = np.ones(len(first))
            for draw in range(k):
                miss *= np.maximum(0, n - sizes - draw) / (n - draw)
            random_values[t] = masses @ (1 - miss) / instance["normalization"]
            iid_reach = q @ np.array([
                1 - (1 - sizes / n) ** m for m in range(1, k + 1)])
            ew_values[t] = masses @ iid_reach / instance["normalization"]
    optimal_set, optimal = fixed_optimum(instance)
    return {"random": random_values, "uniform_ew": ew_values,
            "optimal": optimal,
            "optimal_set": np.asarray(optimal_set, dtype=np.int16)}


def audit(instance):
    rng = np.random.default_rng(71)
    n, k = int(instance["n"]), int(instance["k"])
    for _ in range(100):
        t = int(rng.integers(instance["horizon"]))
        order = rng.permutation(n)
        a_size = int(rng.integers(k + 1))
        b_size = int(rng.integers(a_size, k + 1))
        a, b = order[:a_size], order[:b_size]
        assert -1e-12 <= reward(instance, t, a) <= reward(instance, t, b) + 1e-12
        if len(b) < n:
            candidate = int(order[b_size])
            gain_a = reward(instance, t, np.r_[a, candidate]) - reward(instance, t, a)
            gain_b = reward(instance, t, np.r_[b, candidate]) - reward(instance, t, b)
            assert gain_a + 1e-10 >= gain_b >= -1e-10
    return True
