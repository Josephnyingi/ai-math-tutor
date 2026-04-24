# ASR Pipeline — Reported Metrics

## 1. Number-word parser (extract_integer)

- Overall accuracy: **100.0%** (180/180)

| Language | N | Accuracy |
|---|---|---|
| en | 60 | 100.0% |
| fr | 60 | 100.0% |
| kin | 60 | 100.0% |

## 2. Language detection (lang_detect.detect)

- Overall accuracy: **84.2%** on 120 utterances (30 per class)

| Language | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| en | 30 | 0.730 | 0.900 | 0.806 |
| fr | 30 | 0.962 | 0.833 | 0.893 |
| kin | 30 | 0.920 | 0.767 | 0.836 |
| mix | 30 | 0.812 | 0.867 | 0.839 |

## 3. Pitch-shift normaliser stability

- Semitone shift applied at inference: `-4.5` (child → adult register)

- Energy ratio (output / input) across 0.2–2.0 s sines: **0.993** (target ≈ 1.0)

- NaN / Inf stable across all durations: **True**

## 4. Whisper-tiny WER on real audio

_Run with_ `python3 scripts/eval_asr.py --audio-csv <your.csv>`
where the CSV has columns `audio_path,reference,language`. Recommended eval set: Mozilla Common Voice Kinyarwanda v17 test split filtered to numeric utterances.
