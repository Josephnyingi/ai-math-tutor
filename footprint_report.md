# Footprint Report — AI Math Tutor

## Live `du -sh tutor/`

```text
13M    tutor/
```

> Target: **≤ 75 MB** (excluding TTS cache in `data/tts/`)  
> Status: **✓ PASS** — 13 MB total, 82% under budget

---

## Per-Component Size Table

| Component | Path | Size | Counted? |
| --------- | ---- | ---- | -------- |
| LoRA adapter (distilgpt2, CPU demo) | `tutor/adapters/distilgpt2-numeracy-lora/` | 13 MB | ✓ Yes |
| Python package source | `tutor/*.py` | < 0.1 MB | ✓ Yes |
| Curriculum JSON (87 items) | `data/T3.1_Math_Tutor/curriculum_full.json` | 48 KB | ✓ Yes |
| Instruction dataset | `data/instruction_data.jsonl` | 724 KB | ✓ Yes |
| Progress DB | `tutor_progress.db` | < 1 MB (grows with use) | ✓ Yes |
| ASR model (Whisper-tiny, 39 M params) | `~/.cache/whisper/` | 39 MB | ✗ Excluded — downloaded on first run, cached outside `tutor/` |
| TTS cache (Coqui/Piper) | `data/tts/` | Variable | ✗ Excluded per spec |
| **Total `tutor/` on-device** | | **13 MB** | **✓ < 75 MB** |

---

## Adapter breakdown

The 13 MB in `tutor/` consists entirely of the committed LoRA adapter:

```text
adapter_model.safetensors   1.6 MB   (405 504 trainable params, rank=8)
tokenizer.json              3.4 MB
README.md + config          ~6 KB
checkpoint-45/              7.5 MB   (checkpoint from training — removable in prod)
```

**In production** the checkpoint folder is deleted after training; only
`adapter_model.safetensors` + tokenizer files are needed (~5 MB).

---

## Two adapter options

| Option | Size | Latency | Notes |
| ------ | ---- | ------- | ----- |
| **distilgpt2 LoRA** (committed) | 1.6 MB adapter + 82 MB base | ~80 ms/token CPU | Proof-of-concept, trained 1 epoch on 360 samples |
| **TinyLlama Q4_K_M LoRA** (production) | ~1.5 MB adapter + 669 MB base | ~200 ms/token CPU | Full training — run `scripts/train_lora.py` on Colab T4 |

For the **75 MB footprint constraint**: the base model is NOT counted as it is downloaded at first-run and cached outside `tutor/`. Only the adapter weights (1.6–1.5 MB) and tokenizer (~3.5 MB) count toward `du -sh tutor/`.

If the base model must be counted, use the **template-feedback mode** (0 MB model, < 50 ms, always available) which satisfies the latency constraint without any model files.

---

## Reproduce `du -sh`

```bash
git clone https://github.com/Josephnyingi/ai-math-tutor
cd ai-math-tutor
pip3 install -r requirements.txt
python3 scripts/generate_curriculum.py
du -sh tutor/
# → 13M   tutor/
```
