"""Add the two Wan baselines to the saved paired evaluation streams.

Archive the current results and plots before running this script. Existing
learner actions, rewards, and prefix comparators are left untouched.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from algorithms.wan_baselines import Wan2023, Wan2026
from create_instance import INSTANCE_ROOT, ROOT, load_instance
from main import (POLICY_SEED, SEED_STRIDE, STREAM_SEED_OFFSET, _filename,
                  _sha256, _tune_policy, run_policy)


BASELINES = {"Wan-2023": Wan2023, "Wan-2026": Wan2026}
ARCHIVE_ROOT = ROOT / "archive/2026-10-06_pre_wan_baselines"


def _archive():
    if ARCHIVE_ROOT.exists():
        if not ((ARCHIVE_ROOT / "results").exists()
                and (ARCHIVE_ROOT / "plots").exists()):
            raise FileExistsError("Incomplete Wan-baseline archive")
        return
    ARCHIVE_ROOT.mkdir(parents=True)
    shutil.copytree(ROOT / "results", ARCHIVE_ROOT / "results")
    shutil.copytree(ROOT / "plots", ARCHIVE_ROOT / "plots")


def _write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def add(name):
    output = ROOT / "results" / name
    instance, config = load_instance(INSTANCE_ROOT / name)
    tuning_instance, _ = load_instance(INSTANCE_ROOT / name, tuning=True)
    summary = json.loads((output / "experiment.json").read_text())
    with (output / "runs.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (output / "tuning.csv").open(newline="") as stream:
        tuning_rows = list(csv.DictReader(stream))
    tuning_base = int(config["tuning_parameters"]["seed"])
    evaluation_base = int(config["parameters"]["seed"])
    horizon = int(instance["horizon"])

    for index, (method, policy_class) in enumerate(BASELINES.items(), start=5):
        if method in summary["algorithms"]:
            continue
        seeds = [
            (tuning_base + STREAM_SEED_OFFSET + SEED_STRIDE * repetition,
             POLICY_SEED + tuning_base + SEED_STRIDE * index
             + 100_003 * repetition)
            for repetition in range(int(summary["tuning"]["repetitions"]))]
        candidates, choice = _tune_policy(
            tuning_instance, method, policy_class, seeds, output,
            parameter_horizon=horizon, reuse_existing=True)
        selected = {
            "eta_multiplier": choice["eta_multiplier"],
            "policy_kwargs": choice["policy_kwargs"],
            "tuning_reward": choice["mean_normalized_reward"],
            "candidate": choice["candidate"],
        }
        summary["tuning"]["selected"][method] = selected
        tuning_rows.extend({
            "method": method,
            "candidate": candidate["candidate"],
            "eta_multiplier": candidate["eta_multiplier"],
            "policy_kwargs": json.dumps(candidate["policy_kwargs"], sort_keys=True),
            "mean_normalized_reward": candidate["mean_normalized_reward"],
        } for candidate in candidates)
        print(f"{name}: selected {method} {selected}", flush=True)

        for repetition in range(int(summary["evaluation"]["repetitions"])):
            run_dir = output / "evaluation" / f"run_{repetition:02d}"
            archived_run = ARCHIVE_ROOT / "results" / name / "evaluation" / f"run_{repetition:02d}"
            for filename in ("stream.npz", "references.npz"):
                if _sha256(run_dir / filename) != _sha256(archived_run / filename):
                    raise AssertionError(f"Paired evaluation input changed: {run_dir / filename}")
            with np.load(run_dir / "stream.npz", allow_pickle=False) as saved:
                evaluation_instance = dict(instance)
                evaluation_instance.update({key: saved[key] for key in saved.files})
            stream_seed = (evaluation_base + STREAM_SEED_OFFSET
                           + SEED_STRIDE * repetition)
            policy_seed = (POLICY_SEED + evaluation_base
                           + SEED_STRIDE * index + 100_003 * repetition)
            policy, arrays = run_policy(
                evaluation_instance, policy_class, policy_seed,
                eta_multiplier=selected["eta_multiplier"],
                policy_kwargs=selected["policy_kwargs"])
            data_file = run_dir / f"{_filename(method)}.npz"
            np.savez_compressed(data_file, **arrays)
            row = {
                "method": method,
                "repetition": repetition,
                "stream_seed": stream_seed,
                "policy_seed": policy_seed,
                "mean_normalized_reward": float(arrays["normalized_rewards"].mean()),
                "mean_raw_reward": float(arrays["raw_rewards"].mean()),
                "data_file": str(data_file.relative_to(output)),
                "reference_file": str((run_dir / "references.npz").relative_to(output)),
                "stream_file": str((run_dir / "stream.npz").relative_to(output)),
                "sha256": _sha256(data_file),
            }
            rows.append(row)
            record = dict(row, parameters=policy.parameters,
                          diagnostics=policy.diagnostics())
            (run_dir / f"{_filename(method)}.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n")
            print(f"{name}: {method} repetition {repetition + 1} "
                  f"reward {row['mean_normalized_reward']:.6f}", flush=True)

        summary["algorithms"].append(method)
        summary["evaluation"]["learner_refresh"] = (
            "Wan-2023 and Wan-2026 were added on the saved paired streams; "
            "all five earlier learners and every prefix comparator were reused.")
        summary["tuning"]["wan_parameter_note"] = (
            "Wan-2023 block length and eta multiplier, and Wan-2026 "
            "exploration probability and eta multiplier, are selected on "
            "independent short tuning streams. Their parameter horizon is "
            "the full evaluation horizon. The published finite-horizon "
            "formulas are outside their nondegenerate regimes here.")
        summary["tuning"]["non_learning_rate_parameters"] = (
            "Published values for the five original methods; Wan-2023 block "
            "length and Wan-2026 exploration probability tuned separately")
        (output / "experiment.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n")
        _write_csv(output / "runs.csv", rows, list(rows[0]))
        _write_csv(output / "tuning.csv", tuning_rows,
                   ["method", "candidate", "eta_multiplier", "policy_kwargs",
                    "mean_normalized_reward"])


if __name__ == "__main__":
    _archive()
    for dataset in ("influence", "teamformation", "coverage"):
        add(dataset)
