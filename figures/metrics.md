# AI Math Tutor — Reported Metrics

_Machine_: `Linux-6.8.0-106-generic-aarch64-with-glibc2.35`  
_Python_: `3.10.12`  
_Generated_: `2026-04-24T12:06:27+00:00`

## Headline

| Layer | Metric | Value | 95% CI / notes |
|---|---|---|---|
| **Knowledge Tracing** | BKT AUC (held-out) | **0.5662** | 95% CI [0.5473, 0.5847] |
| Knowledge Tracing | Elo baseline AUC | 0.5170 | Δ = +0.0492 |
| Knowledge Tracing | Prior-only baseline AUC | 0.5000 | (no learning) |
| Knowledge Tracing | Brier score (BKT) | 0.2852 | lower is better |
| Knowledge Tracing | Log-loss (BKT) | 0.7904 | lower is better |
| **ASR number parser** | Accuracy (EN/FR/KIN, n=180) | **100.0%** | 60 per language |
| **Language detection** | Accuracy (n=120) | **84.2%** | 30 each EN/FR/KIN/mix |
| Language detection | en F1 | 0.806 | P=0.730 R=0.900 |
| Language detection | fr F1 | 0.893 | P=0.962 R=0.833 |
| Language detection | kin F1 | 0.836 | P=0.920 R=0.767 |
| Language detection | mix F1 | 0.839 | P=0.812 R=0.867 |
| Pitch-shift normaliser | Energy ratio | 0.993 | target ≈ 1.0, stable |
| **Feedback (template)** | Rubric (0–6) | **5.922** | ≥ 5/6 on 100.0% of 90 cases |
| Feedback (template) | BLEU-2 | 0.931 | vs. 3 gold refs per cell |
| Feedback (template) | ROUGE-L | 0.954 | — |
| Feedback (template) | p95 latency | 0.0 ms | p50 0.0 ms |

## Latency (per-stage, measured on run machine)

| Stage | p50 ms | p95 ms | p99 ms | N |
|---|---|---|---|---|
| lang_detect | 0.0028 | 0.0037 | 0.0045 | 2000 |
| bkt_update | 0.0003 | 0.0004 | 0.0004 | 2000 |
| adaptive_select | 0.0043 | 0.0048 | 0.0062 | 200 |
| visual_grounding (render) | 0.0278 | 0.0519 | 15.7691 | 50 |
| visual_grounding (blob count) | 2.45 | 4.0799 | 4.4522 | 50 |
| template_feedback | 0.0006 | 0.0008 | 0.0011 | 200 |
| end_to_end_scoring | 0.0049 | 0.0058 | 0.0119 | 200 |

## BKT per-skill breakdown

| Skill | N | AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| counting | 738 | 0.57 | 0.5023 | 0.6314 | 0.5595 |
| number_sense | 570 | 0.53 | 0.4129 | 0.5664 | 0.4776 |
| addition | 901 | 0.5771 | 0.4973 | 0.6596 | 0.5671 |
| subtraction | 999 | 0.5611 | 0.5214 | 0.6381 | 0.5738 |
| word_problem | 392 | 0.5758 | 0.5176 | 0.5057 | 0.5116 |

## Reproducibility

Every number in this document can be regenerated with:

```bash
python3 scripts/eval_end_to_end.py
```
which runs `eval_bkt.py`, `eval_asr.py`, `eval_feedback.py` and the latency micro-benchmarks, writing `figures/metrics.json` and this file (`figures/metrics.md`).

### Optional WER on real audio

```bash
python3 scripts/eval_asr.py --audio-csv your_cv_test.csv
```
CSV columns: `audio_path,reference,language` (e.g. Common Voice KIN).