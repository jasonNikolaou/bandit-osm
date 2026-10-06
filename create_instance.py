"""Create every canonical benchmark instance in ``instances/``.

Running this file with no arguments creates the complete experiment suite.  An
existing instance is never replaced unless ``--force`` is supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from datasets.benchmarks import GENERATORS, audit


ROOT = Path(__file__).resolve().parent
INSTANCE_ROOT = ROOT / "instances"
TUNING_HORIZON = 10_000
TUNING_SEED_OFFSET = 10_000_019

EXPERIMENTS = {
    "influence": {
        "kind": "influence", "horizon": 10_000, "seed": 41021,
        "retention": .05, "k": 2,
    },
    "teamformation": {
        "kind": "teamformation", "horizon": 100_000, "seed": 51022,
        "low_fraction": .75, "noise": 1.0, "requirement_spread": .05,
    },
    "coverage": {
        "kind": "coverage", "horizon": 100_000, "seed": 61021,
        "low_fraction": .75, "shared_topics": 3,
        "shared_generalists": 10,
    },
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def create_instance(name, config, *, root=INSTANCE_ROOT, force=False):
    output = Path(root) / name
    archive = output / "instance.npz"
    tuning_archive = output / "tuning_instance.npz"
    metadata = output / "config.json"
    if archive.exists() and tuning_archive.exists() and metadata.exists() and not force:
        return output
    if output.exists() and not force:
        raise FileExistsError(f"Incomplete instance directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    generator = GENERATORS[config["kind"]]
    arguments = {key: value for key, value in config.items() if key != "kind"}
    instance = generator(**arguments)
    tuning_arguments = dict(arguments)
    tuning_arguments["horizon"] = min(TUNING_HORIZON, int(arguments["horizon"]))
    tuning_arguments["seed"] = int(arguments["seed"]) + TUNING_SEED_OFFSET
    tuning_instance = generator(**tuning_arguments)
    audit(instance)
    audit(tuning_instance)
    arrays = {key: value for key, value in instance.items() if key != "kind"}
    tuning_arrays = {
        key: value for key, value in tuning_instance.items() if key != "kind"}
    np.savez_compressed(archive, **arrays)
    np.savez_compressed(tuning_archive, **tuning_arrays)
    record = {
        "name": name,
        "generator": config["kind"],
        "parameters": config,
        "instance_file": archive.name,
        "tuning_parameters": {"kind": config["kind"], **tuning_arguments},
        "tuning_instance_file": tuning_archive.name,
    }
    metadata.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    record["sha256"] = _sha256(archive)
    record["tuning_sha256"] = _sha256(tuning_archive)
    metadata.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return output


def create_all(*, names=None, root=INSTANCE_ROOT, force=False):
    selected = list(names or EXPERIMENTS)
    unknown = sorted(set(selected) - set(EXPERIMENTS))
    if unknown:
        raise ValueError(f"Unknown experiments: {', '.join(unknown)}")
    return [
        create_instance(name, EXPERIMENTS[name], root=root, force=force)
        for name in selected
    ]


def load_instance(directory, *, tuning=False):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    file_key = "tuning_instance_file" if tuning else "instance_file"
    with np.load(directory / config[file_key], allow_pickle=False) as data:
        instance = {key: data[key] for key in data.files}
    instance["kind"] = config["generator"]
    for key in ("n", "k", "horizon", "normalization"):
        if key in instance and np.ndim(instance[key]) == 0:
            instance[key] = instance[key].item()
    return instance, config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", metavar="EXPERIMENT")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for path in create_all(names=args.names or None, force=args.force):
        print(path.relative_to(ROOT))
