import itertools

import numpy as np
import pytest

from datasets.benchmarks import (
    audit, coverage, fixed_optimum, influence, references, reward,
    teamformation,
)


@pytest.fixture(params=["teamformation", "coverage", "influence"])
def instance(request):
    if request.param == "teamformation":
        return teamformation(
            horizon=24, seed=3, n=16, k=3, task_types=3, top_q=3,
            dimension=4, neighbors=3, low_fraction=.5)
    if request.param == "coverage":
        return coverage(
            horizon=24, seed=3, n=16, k=3, task_types=3, topics=3,
            features_per_topic=8, background_features=8, low_fraction=.5)
    return influence(horizon=24, seed=3, k=2, retention=.2)


def test_generators_are_audited(instance):
    assert audit(instance)
    assert all(0 <= reward(instance, t, []) <= 1 for t in range(5))


def test_team_has_low_values_and_nonzero_submodular_overlap():
    data = teamformation(horizon=12, seed=9, n=16, k=3, task_types=3,
                         top_q=3, dimension=4, neighbors=3, low_fraction=.5)
    assert len(data["low_elements"]) == 8
    assert np.any(data["penalties"] < 0)
    assert np.max(data["weights"][:, data["low_mask"]]) <= .005 * 100


def test_team_can_generate_related_task_requirements():
    data = teamformation(
        horizon=12, seed=9, n=16, k=3, task_types=3, top_q=3,
        dimension=4, neighbors=3, low_fraction=.5,
        requirement_spread=.05)
    similarities = data["requirements"] @ data["common_requirement"]
    assert np.all(similarities > .99)
    assert data["requirement_spread"] == pytest.approx(.05)


def test_coverage_has_weak_pool_and_exact_random_reference():
    data = coverage(
        horizon=8, seed=4, n=16, k=3, task_types=2, topics=2,
        features_per_topic=5, background_features=8, low_fraction=.5)
    assert data["low_mask"].sum() == 8
    exact = references(data)["random"]
    teams = list(itertools.combinations(range(data["n"]), data["k"]))
    for t in range(data["horizon"]):
        assert exact[t] == pytest.approx(
            np.mean([reward(data, t, team) for team in teams]))


def test_coverage_can_generate_related_tasks_and_shared_generalists():
    data = coverage(
        horizon=8, seed=4, n=20, k=3, task_types=3, topics=3,
        features_per_topic=5, background_features=8, low_fraction=.25,
        shared_topics=2, shared_generalists=2,
        shared_features_per_topic=3)
    assert data["shared_topics"] == 2
    assert len(data["shared_generalist_elements"]) == 2
    assert np.all(
        data["coverage"][data["shared_generalist_elements"]].sum(axis=1)
        == 2 * 3 + 2)
    assert np.all(data["specific_topics"] >= 2)


def test_fixed_optimum_matches_brute_force(instance):
    chosen, values = fixed_optimum(instance)
    assert len(chosen) == instance["k"]
    assert len(np.unique(chosen)) == instance["k"]
    np.testing.assert_allclose(
        values, [reward(instance, t, chosen) for t in range(instance["horizon"])])
    brute_force = max(
        sum(reward(instance, t, subset) for t in range(instance["horizon"]))
        for subset in itertools.combinations(range(instance["n"]), instance["k"]))
    assert values.sum() == pytest.approx(brute_force)
