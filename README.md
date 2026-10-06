# Efficient Online Submodular Maximization with Bandit Feedback

## Abstract

We study online monotone submodular maximization under a cardinality constraint
with bandit feedback. On each round, the learner selects at most $k$ elements
from a ground set of size $n$ and observes only the value of the selected set.
We develop an efficient algorithm achieving
$\mathcal{O}(k\sqrt{nT\log(n/k)})$ expected $(1-1/e)$-regret over $T$ rounds.
To our knowledge, this is the first polynomial-time algorithm with $\sqrt{T}$
dependence on the horizon for arbitrary monotone submodular functions in the
bandit setting. Our approach uses randomized sampling and a carefully
constructed potential to reduce online submodular maximization to online
linear optimization over the probability simplex.

## Reproduce the experiments

The repository contains the three input instances used in the paper.

| Dataset | $n$ | $k$ | $T$ |
| --- | ---: | ---: | ---: |
| `influence` | 34 | 2 | 10,000 |
| `teamformation` | 100 | 3 | 100,000 |
| `coverage` | 100 | 3 | 100,000 |

Install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run all algorithms on all datasets:

```bash
python main.py
```

This runs Sub-EW, Random, MSE3, OpaqueGreedy, and Bandit-IG with 10 evaluation
runs and 3 separate tuning runs. To change these counts, use:

```bash
python main.py --repetitions 10 --tuning-repetitions 3
```

Results are saved in `results/`. To reproduce the PDF figures, run:

```bash
python plot.py
```

The plots show the mean over the 10 runs, with shaded regions equal to one
sample standard deviation. The y-axis reports average approximate regret:

```text
AAR(t) = ((1 - 1/e) * cumulative optimal fixed-set reward
          - cumulative algorithm reward) / t.
```

The comparator is the exact best fixed cardinality-$k$ set in hindsight,
computed separately for each evaluation stream by exhaustive enumeration. It
is used only for evaluation and is not revealed to the algorithms.

To run or plot one dataset, pass its name:

```bash
python main.py influence
python plot.py influence
```

## Files

- `create_instance.py`: creates the three instances.
- `main.py`: runs and saves the experiments.
- `plot.py`: creates the three figures.
- `algorithms/`: the five algorithms used in the experiments.
- `datasets/benchmarks.py`: the three objective functions.
- `instances/`: the three paper instances.
- `tests/`: correctness and end-to-end tests.

Run the tests with:

```bash
python -m pytest -q
```
