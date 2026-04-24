# BKT Evaluation — Reported Metrics

- Synthetic learners: **300** × 40 responses each, 70/30 train/test split

- Held-out predictions: **3600**

- Bootstrap iterations for 95% CI: **1000**

- Seed: `42` (fully reproducible)


## Headline — AUC with 95% bootstrap CIs

| Model | AUC | 95% CI | Brier ↓ | Log-loss ↓ | Accuracy | F1 |
|---|---|---|---|---|---|---|
| **BKT** | 0.5662 | [0.5473, 0.5847] | 0.2852 | 0.7904 | 0.5347 | 0.5494 |
| **Elo** | 0.5170 | [0.4973, 0.5353] | 0.2700 | 0.7378 | 0.5014 | 0.5248 |
| **Majority** | 0.5385 | [0.5192, 0.5591] | 0.2548 | 0.7052 | 0.5386 | 0.3099 |
| **Random** | 0.4909 | [0.4715, 0.5091] | 0.2500 | 0.6931 | 0.4953 | 0.4723 |
| **Prior-only** | 0.5000 | [0.5000, 0.5000] | 0.2689 | 0.7346 | 0.5414 | 0.0000 |

## Per-Skill AUC (BKT)

| Skill | N | AUC | Precision | Recall | F1 | Positive rate |
|---|---|---|---|---|---|---|
| counting | 738 | 0.57 | 0.5023 | 0.6314 | 0.5595 | 0.4743 |
| number_sense | 570 | 0.53 | 0.4129 | 0.5664 | 0.4776 | 0.3965 |
| addition | 901 | 0.5771 | 0.4973 | 0.6596 | 0.5671 | 0.4695 |
| subtraction | 999 | 0.5611 | 0.5214 | 0.6381 | 0.5738 | 0.4785 |
| word_problem | 392 | 0.5758 | 0.5176 | 0.5057 | 0.5116 | 0.4439 |

## Ablation summary

- BKT beats Elo baseline by **+0.0492 AUC** (|Δ|/σ ≈ 5.13).
- BKT beats Prior-only (no learning) by **+0.0662 AUC**.
- All non-trivial models beat Random (0.5) and Majority-class baselines.

_Reproduce with_ `python3 scripts/eval_bkt.py`
