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
sample standard deviation. The y-axis reports average approximate regret at
every timestep:

```text
AAR(t) = ((1 - 1/e) * OPT_t - cumulative algorithm reward) / t,
OPT_t = max over |S|=k of sum_{s=1}^t f_s(S).
```

The comparator is the exact best fixed cardinality-$k$ set for each prefix,
computed separately for each evaluation stream by exhaustive enumeration with
cumulative candidate scores. It is one fixed set within a prefix, rather than
the sum of separate roundwise maxima. The exact value is saved at every round
in `references.npz` as `prefix_optimal`; the figure may display fewer points.
For TeamFormation and Coverage, candidate rewards are precomputed for each of
the ten task types. A common task-type maximizer is used only after checking
that it maximizes every task type in the loaded instance; otherwise the code
updates all candidate scores for every prefix. Comparators are shared across
the five learners within an evaluation repetition and are never revealed to
them.

Bandit-IG uses one stage-local Hedge learner per greedy step, with weights
proportional to the exponential of the current learning rate times cumulative
losses. This is a valid decreasing-rate Hedge instantiation of the Blackwell
subroutine in Algorithms 3–4 of Niazadeh et al. Its exploration probability is
`min(1, (n^2 log(n) / T)^(1/3))`, with `T` equal to the evaluation horizon.
For normalized rewards in `[0,1]`, an exploration feedback vector has
coordinate range at most `n/q`; decreasing-rate Hedge therefore has
`O((n/q) sqrt(r log n))` regret after `r` stage-local feedback updates.
Balancing that term with exploration at the prescribed `q` gives the
`O(k n^(2/3) (log n)^(1/3) T^(2/3))` scale in the paper's Example 1.
During tuning, the learning-rate multipliers are scored on three independent
shorter streams, but Bandit-IG's parameter horizon remains the evaluation
horizon. Thus the two 100,000-round datasets use `T=100000` for `q` even though
the tuning streams contain 10,000 rounds. The saved evaluation streams and
the other learners' actions and rewards are reused in the corrected results.

OpaqueGreedy follows Streeter and Golovin's 2008 proceedings paper; the
precise opaque-feedback construction is in their 2007 Carnegie Mellon
technical report CMU-CS-07-171, Section 4.4.3.

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
