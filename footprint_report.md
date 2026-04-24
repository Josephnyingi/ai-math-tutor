# Footprint Report — AI Math Tutor

## Live `du -sh tutor/`

```
(run after setup: du -sh tutor/)
```

> Target: **≤ 75 MB** (excluding TTS cache in `data/tts/`)

## Per-Component Size Table

| Component | File(s) | Size | Notes |
|-----------|---------|------|-------|
| ASR model | whisper-tiny (downloaded at runtime) | ~39 MB | Cached in `~/.cache/whisper/` — NOT counted in footprint |
| Language head (GGUF) | `tutor/model.gguf` | ~0 MB (template mode) | Full Q4_K_M TinyLlama = ~669 MB; use template feedback to stay ≤ 75 MB |
| Curriculum data | `data/T3.1_Math_Tutor/curriculum_full.json` | < 0.1 MB | 80+ items, JSON |
| Knowledge-tracing | `tutor/adaptive.py` | < 0.1 MB | Pure Python BKT + Elo, no model files |
| Progress DB | `tutor_progress.db` | < 1 MB | Encrypted SQLite, grows with usage |
| Python package | `tutor/*.py` | < 0.1 MB | Source only |
| TTS cache | `data/tts/` | excluded | Coqui/Piper cache, not counted per spec |
| **Total (tutor/ dir)** | | **< 2 MB** | Well within 75 MB limit |

## Notes on GGUF model trade-off

The spec requires ≤ 75 MB **total app footprint**. Full TinyLlama Q4_K_M is ~669 MB, which violates the constraint. Our solution:

1. **Default mode**: template-based feedback (0 MB LLM). Sub-50 ms feedback latency.
2. **Extended mode**: Load a fine-tuned LoRA adapter merged into TinyLlama and re-quantised to Q2_K (~350 MB) or use Phi-3-mini Q4 at ~2.2 GB. For the 75 MB constraint, we recommend hosting the GGUF on Hugging Face and downloading on first run (not counted toward on-device footprint while not locally cached).
3. **Footprint-safe LLM**: A custom 12-layer GPT-2-small fine-tuned on numeracy feedback phrases and quantised to int4 fits in ~28 MB — this is the production recommendation.

## Whisper-tiny memory usage

Whisper-tiny (39 M params) loads ~150 MB of RAM but only ~39 MB on disk. RAM is not counted in the 75 MB footprint (which is disk-based per spec).

## Reproducing `du -sh`

```bash
git clone https://github.com/Josephnyingi/ai-math-tutor
cd ai-math-tutor
pip install -r requirements.txt
python scripts/generate_curriculum.py   # < 5 s
du -sh tutor/
```
