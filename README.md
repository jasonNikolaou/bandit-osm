# Efficient Online Submodular Maximization with Bandit Feedback

Code, fixed benchmark instances, and figures for the experiments on online
submodular maximization with bandit feedback.

## Experiments

| Dataset | Ground set size | Set size | Rounds |
| --- | ---: | ---: | ---: |
| Influence | 34 | 2 | 10,000 |
| TeamFormation | 100 | 3 | 100,000 |
| Coverage | 100 | 3 | 100,000 |

The compared algorithms are:

- Sub-EW — this work
- Random
- MSE3 — [Pasteris et al. (2024)](https://proceedings.mlr.press/v238/pasteris24a.html)
- OpaqueGreedy — [Streeter and Golovin (2008)](https://proceedings.neurips.cc/paper/2008/hash/5751ec3e9a4feab575962e78e006250d-Abstract.html)
- Bandit-IG — [Niazadeh et al. (2023)](https://arxiv.org/abs/2102.11050)
- BanditMLSM — [Wan et al. (2023)](https://proceedings.mlr.press/v202/wan23e.html)
- Poisson-BE — [Wan and Zhang (2026)](https://arxiv.org/abs/2608.24627)

Figures show mean average approximate regret over 10 evaluation repetitions,
with one sample standard deviation shaded. At each round, the benchmark is the
best fixed set for the stream prefix.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
python plot.py
```

`main.py` uses the included instances and writes local results to `results/`.
That directory is generated and is not included in the repository. `plot.py`
reads those results to regenerate the committed figures in `plots/`.

To run or plot a single dataset, pass `influence`, `teamformation`, or
`coverage` as an argument. Use `python main.py --help` for other options.
