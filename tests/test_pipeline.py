import json
import csv

import numpy as np

from algorithms import POLICIES
from create_instance import create_instance, load_instance
from main import _tuning_configs, run_experiment
from plot import plot_experiment


def test_complete_saved_pipeline(tmp_path):
    instance_root = tmp_path / "instances"
    result_root = tmp_path / "results"
    plot_root = tmp_path / "plots"
    name = "tiny"
    config = {
        "kind": "coverage", "horizon": 12, "seed": 7,
        "n": 16, "k": 3, "task_types": 2, "topics": 2,
        "features_per_topic": 5, "background_features": 8,
        "low_fraction": .5,
    }
    create_instance(name, config, root=instance_root)
    instance, metadata = load_instance(instance_root / name)
    assert instance["horizon"] == 12
    assert metadata["parameters"] == config
    tuning, _ = load_instance(instance_root / name, tuning=True)
    assert tuning["horizon"] == 12

    run_experiment(
        name, instance_root=instance_root, result_root=result_root,
        evaluation_repetitions=3, tuning_repetitions=2)
    summary = json.loads((result_root / name / "experiment.json").read_text())
    assert summary["algorithms"] == list(POLICIES)
    assert set(summary["tuning"]["selected"]) == set(POLICIES)
    assert summary["evaluation"]["repetitions"] == 3
    assert summary["evaluation"]["paired_streams"] is True
    assert (result_root / name / "tuning.csv").exists()
    with (result_root / name / "tuning.csv").open(newline="") as stream:
        tuning_rows = list(csv.DictReader(stream))
    assert "exploration_multiplier" not in tuning_rows[0]
    assert sum(row["method"] == "Random" for row in tuning_rows) == 1
    for method in set(POLICIES) - {"Random"}:
        assert sum(row["method"] == method for row in tuning_rows) == len(
            _tuning_configs(method, tuning))
    with (result_root / name / "runs.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3 * len(POLICIES)
    for row in rows:
        with np.load(result_root / name / row["data_file"]) as data:
            assert set(data.files) == {
                "normalized_rewards", "raw_rewards", "actions",
                "action_sizes", "iterates",
            }
            assert len(data["raw_rewards"]) == 12
    final = plot_experiment(name, result_root=result_root, plot_root=plot_root)
    assert set(final) == set(POLICIES)
    assert (result_root / name / "variability_bands.npz").exists()
    assert [path.name for path in plot_root.iterdir()] == [
        "tiny.pdf"]
