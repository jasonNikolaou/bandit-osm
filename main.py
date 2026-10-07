"""Run the Influence, Team Formation, and Coverage experiments."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from algorithms import POLICIES
from create_instance import EXPERIMENTS, INSTANCE_ROOT, ROOT, create_all, load_instance
from datasets.benchmarks import (audit, candidate_task_rewards, references,
                                 resample_stream, reward)


RESULT_ROOT = ROOT / "results"
POLICY_SEED = 73_000
EVALUATION_REPETITIONS = 10
TUNING_REPETITIONS = 3
SEED_STRIDE = 1009
STREAM_SEED_OFFSET = 20_000_033
ETA_MULTIPLIERS = (1.0, 0.25, 4.0, 16.0, 64.0)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _filename(method):
    return re.sub(r"[^a-z0-9]+", "_", method.lower()).strip("_")


def run_policy(instance, policy_class, seed, *, eta_multiplier=1.0,
               parameter_horizon=None, policy_kwargs=None):
    horizon = int(instance["horizon"])
    k = int(instance["k"])
    policy = policy_class(
        instance["n"], k,
        horizon if parameter_horizon is None else int(parameter_horizon),
        seed, eta_multiplier=eta_multiplier, **(policy_kwargs or {}))
    initial = np.asarray(policy.iterate(), dtype=np.float32)
    iterates = np.empty((horizon,) + initial.shape, dtype=np.float32)
    normalized = np.empty(horizon)
    actions = np.full((horizon, k), -1, dtype=np.int16)
    action_sizes = np.empty(horizon, dtype=np.int8)
    for t in range(horizon):
        action = np.sort(np.asarray(policy.act(), dtype=np.int16))
        value = reward(instance, t, action)
        policy.observe(value)
        normalized[t] = value
        actions[t, :len(action)] = action
        action_sizes[t] = len(action)
        iterates[t] = policy.iterate()
    return policy, {
        "normalized_rewards": normalized,
        "raw_rewards": normalized * float(instance["normalization"]),
        "actions": actions,
        "action_sizes": action_sizes,
        "iterates": iterates,
    }


def _tuning_grid(method):
    return (1.0,) if method == "Random" else ETA_MULTIPLIERS


def _tuning_configs(method, instance):
    """Separate practical grids for the two Wan algorithms."""
    if method == "BanditMLSM":
        multipliers = ((1e4, 1e5, 1e6, 1e7)
                       if int(instance["n"]) < 50
                       else (1e6, 1e7, 1e8))
        return [
            {"eta_multiplier": multiplier,
             "policy_kwargs": {"block_size": block}}
            for block in (8, 32, 128, 512) for multiplier in multipliers]
    if method == "Poisson-BE":
        return [
            {"eta_multiplier": multiplier,
             "policy_kwargs": {"exploration_probability": gamma}}
            for gamma in (.05, .2, .5)
            for multiplier in (.25, 1.0, 4.0, 16.0)] + [
            {"eta_multiplier": multiplier,
             "policy_kwargs": {"exploration_probability": gamma}}
            for gamma in (.05, .2, .5)
            for multiplier in (64.0, 256.0)]
    return [{"eta_multiplier": multiplier, "policy_kwargs": {}}
            for multiplier in _tuning_grid(method)]


def _stream_arrays(instance):
    if instance["kind"] == "influence":
        return {key: instance[key] for key in ("labels", "masses", "sizes")}
    return {"task_ids": instance["task_ids"]}


def _tune_policy(instance, method, policy_class, seeds, output,
                 *, parameter_horizon=None, reuse_existing=False):
    directory = output / "tuning" / _filename(method)
    directory.mkdir(parents=True, exist_ok=True)
    candidates = []
    for candidate, configuration in enumerate(_tuning_configs(method, instance)):
        eta_multiplier = configuration["eta_multiplier"]
        policy_kwargs = configuration["policy_kwargs"]
        record_file = directory / f"candidate_{candidate:02d}.json"
        if reuse_existing and record_file.exists():
            record = json.loads(record_file.read_text())
            if (record["eta_multiplier"] == eta_multiplier
                    and record.get("policy_kwargs", {}) == policy_kwargs
                    and [(run["stream_seed"], run["policy_seed"])
                         for run in record["runs"]] == list(seeds)):
                candidates.append(record)
                continue
        runs = []
        for repetition, (stream_seed, policy_seed) in enumerate(seeds):
            tuning_stream = resample_stream(instance, stream_seed)
            policy, arrays = run_policy(
                tuning_stream, policy_class, policy_seed,
                eta_multiplier=eta_multiplier,
                parameter_horizon=parameter_horizon,
                policy_kwargs=policy_kwargs)
            archive = directory / (
                f"candidate_{candidate:02d}_run_{repetition:02d}.npz")
            np.savez_compressed(archive, **arrays)
            runs.append({
                "repetition": repetition,
                "stream_seed": stream_seed,
                "policy_seed": policy_seed,
                "mean_normalized_reward": float(
                    arrays["normalized_rewards"].mean()),
                "parameters": policy.parameters,
                "diagnostics": policy.diagnostics(),
                "data_file": str(archive.relative_to(output)),
                "sha256": _sha256(archive),
            })
        record = {
            "method": method,
            "candidate": candidate,
            "eta_multiplier": eta_multiplier,
            "policy_kwargs": policy_kwargs,
            "mean_normalized_reward": float(np.mean([
                run["mean_normalized_reward"] for run in runs])),
            "runs": runs,
        }
        record_file.write_text(
            json.dumps(record, default=_jsonable, indent=2, sort_keys=True)
            + "\n")
        candidates.append(record)
    return candidates, max(
        candidates, key=lambda item: item["mean_normalized_reward"])


def _archive_existing(output):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_root = Path(tempfile.gettempdir()) / "bandit-osm-superseded-results"
    archive_root.mkdir(parents=True, exist_ok=True)
    destination = archive_root / f"{output.name}__{stamp}"
    counter = 1
    while destination.exists():
        destination = archive_root / f"{output.name}__{stamp}_{counter}"
        counter += 1
    shutil.move(str(output), str(destination))
    return destination


def run_experiment(name, *, instance_root=INSTANCE_ROOT,
                   result_root=RESULT_ROOT, force=False,
                   evaluation_repetitions=EVALUATION_REPETITIONS,
                   tuning_repetitions=TUNING_REPETITIONS):
    instance_dir = Path(instance_root) / name
    instance, config = load_instance(instance_dir)
    tuning_instance, _ = load_instance(instance_dir, tuning=True)
    audit(instance)
    audit(tuning_instance)

    output = Path(result_root) / name
    if (output / "experiment.json").exists() and not force:
        print(f"skip {name}: results already exist")
        return output
    if output.exists():
        if not force:
            raise FileExistsError(f"Incomplete result directory: {output}")
        print(f"archived previous results at {_archive_existing(output)}")
    output.mkdir(parents=True)

    selected = {}
    tuning_rows = []
    tuning_base = int(config["tuning_parameters"]["seed"])
    for index, (method, policy_class) in enumerate(POLICIES.items()):
        seeds = [
            (tuning_base + STREAM_SEED_OFFSET + SEED_STRIDE * repetition,
             POLICY_SEED + tuning_base + SEED_STRIDE * index
             + 100_003 * repetition)
            for repetition in range(tuning_repetitions)
        ]
        candidates, choice = _tune_policy(
            tuning_instance, method, policy_class, seeds, output,
            parameter_horizon=(int(instance["horizon"])
                               if method in ("Bandit-IG", "BanditMLSM",
                                             "Poisson-BE") else None))
        tuning_rows.extend(candidates)
        selected[method] = {
            "eta_multiplier": choice["eta_multiplier"],
            "policy_kwargs": choice["policy_kwargs"],
            "tuning_reward": choice["mean_normalized_reward"],
            "candidate": choice["candidate"],
        }
        print(
            f"{name}: selected {method} eta x{choice['eta_multiplier']:g} "
            f"(tuning reward {choice['mean_normalized_reward']:.6f})")

    tuning_fields = (
        "method", "candidate", "eta_multiplier", "policy_kwargs",
        "mean_normalized_reward")
    with (output / "tuning.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=tuning_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {**{key: row[key] for key in tuning_fields
                if key != "policy_kwargs"},
             "policy_kwargs": json.dumps(row["policy_kwargs"], sort_keys=True)}
            for row in tuning_rows)

    rows = []
    evaluation_root = output / "evaluation"
    evaluation_root.mkdir()
    evaluation_seed = int(config["parameters"]["seed"])
    task_rewards = candidate_task_rewards(instance)
    for repetition in range(evaluation_repetitions):
        run_dir = evaluation_root / f"run_{repetition:02d}"
        run_dir.mkdir()
        stream_seed = (
            evaluation_seed + STREAM_SEED_OFFSET + SEED_STRIDE * repetition)
        evaluation_instance = resample_stream(instance, stream_seed)
        reference = references(evaluation_instance, task_rewards=task_rewards)
        stream_file = run_dir / "stream.npz"
        reference_file = run_dir / "references.npz"
        np.savez_compressed(stream_file, **_stream_arrays(evaluation_instance))
        np.savez_compressed(reference_file, **reference)

        for index, (method, policy_class) in enumerate(POLICIES.items()):
            policy_seed = (
                POLICY_SEED + evaluation_seed + SEED_STRIDE * index
                + 100_003 * repetition)
            policy, arrays = run_policy(
                evaluation_instance, policy_class, policy_seed,
                eta_multiplier=selected[method]["eta_multiplier"],
                policy_kwargs=selected[method]["policy_kwargs"])
            archive = run_dir / f"{_filename(method)}.npz"
            np.savez_compressed(archive, **arrays)
            record = {
                "method": method,
                "repetition": repetition,
                "stream_seed": stream_seed,
                "policy_seed": policy_seed,
                "mean_normalized_reward": float(
                    arrays["normalized_rewards"].mean()),
                "mean_raw_reward": float(arrays["raw_rewards"].mean()),
                "parameters": policy.parameters,
                "diagnostics": policy.diagnostics(),
                "data_file": str(archive.relative_to(output)),
                "reference_file": str(reference_file.relative_to(output)),
                "stream_file": str(stream_file.relative_to(output)),
                "sha256": _sha256(archive),
            }
            (run_dir / f"{_filename(method)}.json").write_text(
                json.dumps(record, default=_jsonable, indent=2,
                           sort_keys=True) + "\n")
            rows.append(record)
            print(
                f"{name}: run {repetition + 1}/{evaluation_repetitions} "
                f"{method} mean reward "
                f"{record['mean_normalized_reward']:.6f}", flush=True)

    run_fields = (
        "method", "repetition", "stream_seed", "policy_seed",
        "mean_normalized_reward", "mean_raw_reward", "data_file",
        "reference_file", "stream_file", "sha256")
    with (output / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=run_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in run_fields} for row in rows)

    summary = {
        "name": name,
        "instance": config,
        "algorithms": list(POLICIES),
        "tuning": {
            "selection_metric": "highest mean normalized reward",
            "separate_instance": True,
            "eta_multipliers": list(ETA_MULTIPLIERS),
            "eta_multipliers_scope": (
                "shared grid for Sub-EW, MSE3, OpaqueGreedy, and Bandit-IG; "
                "method-specific candidate configurations are in tuning.csv"),
            "non_learning_rate_parameters": (
                "Published values except practical BanditMLSM block length and "
                "Poisson-BE exploration probability, tuned on separate streams"),
            "practical_parameter_note": (
                "BanditMLSM block length and eta multiplier, and Poisson-BE "
                "exploration probability and eta multiplier, are selected "
                "on independent short streams using the evaluation horizon "
                "for parameter formulas. BanditMLSM's eta multiplier scales "
                "a practical reference from Algorithm 1 / Theorem 3.3, not "
                "the product-simplex wrapper rate in Lemma 5.1. The wrapper's "
                "prescribed block length is below one round here."),
            "selected": selected,
            "repetitions": tuning_repetitions,
            "bandit_ig_parameter_horizon": int(instance["horizon"]),
            "bandit_ig_tuning_rounds": int(tuning_instance["horizon"]),
            "bandit_ig_horizon_note": (
                "Bandit-IG q and its learning-rate schedule use the evaluation "
                "horizon; candidate rewards use independent shorter tuning streams."),
        },
        "evaluation": {
            "repetitions": evaluation_repetitions,
            "paired_streams": True,
            "plot_band": (
                "mean plus/minus one sample standard deviation across runs"),
            "structural_instance_fixed": True,
        },
        "reference": {
            "meaning": "exact best fixed cardinality-k set for every stream prefix",
        },
        "normalization": float(instance["normalization"]),
        "horizon": int(instance["horizon"]),
        "cardinality": int(instance["k"]),
    }
    (output / "experiment.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", metavar="DATASET")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--repetitions", type=int, default=EVALUATION_REPETITIONS,
        help="paired evaluation repetitions (default: 10)")
    parser.add_argument(
        "--tuning-repetitions", type=int, default=TUNING_REPETITIONS,
        help="separate tuning repetitions (default: 3)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    names = args.names or list(EXPERIMENTS)
    unknown = sorted(set(names) - set(EXPERIMENTS))
    if unknown:
        raise SystemExit(f"Unknown datasets: {', '.join(unknown)}")
    create_all(names=names)
    for name in names:
        run_experiment(
            name, force=args.force,
            evaluation_repetitions=args.repetitions,
            tuning_repetitions=args.tuning_repetitions)
