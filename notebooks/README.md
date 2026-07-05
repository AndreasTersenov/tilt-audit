# The guided tour

Four notebooks walk through the whole project in order, results and code
together. Each runs top to bottom on CPU from the tracked results files.

| Notebook | What it covers |
|---|---|
| `01_the_bench.ipynb` | The problem, the exact construction, and the null-gate discipline, with a live exactness demonstration |
| `02_sampler_anatomy.ipynb` | Every steering scheme measured against exact targets, and the misspecification interaction including the compensation trap |
| `03_certificates_on_trial.ipynb` | Both proposed runtime certificates under trial: journey accounting and score-based certification, with the three measured verdicts |
| `04_gold_standards_and_transfer.ipynb` | The nonlinear substrate, cheap MCMC gold standards, the transfer decay law, and the honest scope of budget-doubling checks |

Regenerate after new results with
`python notebooks/build_notebooks.py` followed by
`jupyter nbconvert --execute --inplace notebooks/0*.ipynb`.
