# Feedback Generation — Reported Metrics

Held-out set: **90 items** = 3 languages × 2 conditions × 15 answers; 3 gold references per cell.

## Summary by mode

| Mode | Rubric (0–6) | ≥ 5/6 pass rate | BLEU-2 | ROUGE-L | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| **template** | 5.92 | 100.0% | 0.931 | 0.954 | 0.0 | 0.0 |

### template — per condition

| Cell | N | Rubric | BLEU-2 | ROUGE-L | Lang-match | Answer incl. (wrong) |
|---|---|---|---|---|---|---|
| en_correct | 15 | 6.00 | 0.774 | 0.844 | 100% | n/a |
| en_wrong | 15 | 6.00 | 1.000 | 1.000 | 100% | 1.0 |
| fr_correct | 15 | 5.53 | 0.936 | 0.964 | 53% | n/a |
| fr_wrong | 15 | 6.00 | 1.000 | 1.000 | 100% | 1.0 |
| kin_correct | 15 | 6.00 | 0.877 | 0.917 | 100% | n/a |
| kin_wrong | 15 | 6.00 | 1.000 | 1.000 | 100% | 1.0 |
