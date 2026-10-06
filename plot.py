"""Recreate all experiment figures from data stored in ``results/``."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bandit-osm-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from create_instance import EXPERIMENTS, ROOT


RESULT_ROOT = ROOT / "results"
PLOT_ROOT = ROOT / "plots"
ALPHA = 1 - 1 / np.e
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 14
PLOT_POINTS = 2_000
COLORS = {
    "EW": "#0072B2", "Random": "#777777", "MSE3": "#D55E00",
    "OpaqueGreedy": "#009E73", "Bandit-IG": "#CC79A7",
}


def _curve(comparator, rewards):
    rounds = np.arange(1, len(rewards) + 1)
    return (ALPHA * np.cumsum(comparator) - np.cumsum(rewards)) / rounds


def plot_experiment(name, *, result_root=RESULT_ROOT, plot_root=PLOT_ROOT):
    source = Path(result_root) / name
    summary = json.loads((source / "experiment.json").read_text())
    with (source / "runs.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    final = {}
    statistics = {}
    for method in summary["algorithms"]:
        method_rows = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["repetition"]))
        curves = []
        for row in method_rows:
            with np.load(source / row["reference_file"]) as data:
                greedy = data["greedy"]
            with np.load(source / row["data_file"]) as data:
                curves.append(_curve(greedy, data["normalized_rewards"]))
        curves = np.asarray(curves)
        count = len(curves)
        mean = curves.mean(axis=0)
        if count < 2:
            standard_deviation = np.zeros_like(mean)
        else:
            standard_deviation = curves.std(axis=0, ddof=1)
        lower = mean - standard_deviation
        upper = mean + standard_deviation
        final[method] = float(mean[-1])
        statistics[f"{method}_mean"] = mean
        statistics[f"{method}_standard_deviation"] = standard_deviation
        statistics[f"{method}_lower"] = lower
        statistics[f"{method}_upper"] = upper
        rounds = np.arange(1, len(mean) + 1)
        shown = np.unique(np.linspace(
            0, len(mean) - 1, min(PLOT_POINTS, len(mean)), dtype=int))
        axis.fill_between(
            rounds[shown], lower[shown], upper[shown],
            color=COLORS[method], alpha=.16, linewidth=0)
        axis.plot(rounds[shown], mean[shown],
                  color=COLORS[method], linewidth=1.5, label=method)
    axis.axhline(0, color="black", linewidth=.7, alpha=.5)
    axis.set_xlabel("Round $t$", fontsize=AXIS_LABEL_SIZE)
    axis.set_ylabel(r"$\mathrm{AAR}(t)$", fontsize=AXIS_LABEL_SIZE)
    axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    axis.grid(alpha=.2)
    axis.legend(ncol=2, frameon=False)
    figure.subplots_adjust(left=.15, bottom=.17, right=.94, top=.97)
    target = Path(plot_root)
    target.mkdir(parents=True, exist_ok=True)
    output = target / f"{name}.pdf"
    if output.exists():
        archive = Path(tempfile.gettempdir()) / "bandit-osm-superseded-plots"
        archive.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = archive / f"{output.stem}__{stamp}{output.suffix}"
        counter = 1
        while destination.exists():
            destination = archive / (
                f"{output.stem}__{stamp}_{counter}{output.suffix}")
            counter += 1
        shutil.move(str(output), str(destination))
    figure.savefig(output)
    plt.close(figure)
    np.savez_compressed(source / "variability_bands.npz", **statistics)
    (source / "variability_bands.json").write_text(json.dumps({
        "band": "mean plus/minus one sample standard deviation across runs",
        "repetitions": int(summary["evaluation"]["repetitions"]),
        "final": {
            method: {
                "mean": float(statistics[f"{method}_mean"][-1]),
                "standard_deviation": float(
                    statistics[f"{method}_standard_deviation"][-1]),
                "lower": float(statistics[f"{method}_lower"][-1]),
                "upper": float(statistics[f"{method}_upper"][-1]),
            }
            for method in summary["algorithms"]
        },
    }, indent=2, sort_keys=True) + "\n")
    return final


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", metavar="EXPERIMENT")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    discovered = ([] if not RESULT_ROOT.exists() else [
        path.name for path in RESULT_ROOT.iterdir()
        if path.is_dir() and (path / "experiment.json").exists()])
    names = args.names or discovered
    if not names:
        raise SystemExit("No canonical results found. Run python main.py first.")
    index = {name: plot_experiment(name) for name in names}
    (PLOT_ROOT / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(names)} plots to {PLOT_ROOT.relative_to(ROOT)}")
