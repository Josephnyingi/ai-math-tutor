# Process Log — S2.T3.1 AI Math Tutor

**Candidate:** Joseph Nyingi  
**Date:** 2026-04-24  
**Total time:** ~7 hours (initial build 4 h + post-build hardening 3 h)

---

## Hour-by-Hour Timeline

### Hour 1 (0:00–1:00) — Architecture & Data

- Read full brief twice; annotated all 7 technical tasks and scoring rubric.
- Analysed seed files: `curriculum_seed.json` (12 items × 5 skills), `diagnostic_probes_seed.csv`, `child_utt_sample_seed.csv`, `parent_report_schema.json`, `child_utt_index.md`.
- Decided on BKT over DKT (simpler, interpretable, faster at inference, better for the evaluation timeline).
- Chose Whisper-tiny (39 M params, CPU-only) as ASR backbone.
- Decided against loading GGUF at demo time to meet 75 MB constraint — template fallback with LoRA merge as optional upgrade path.
- Created full project structure and scaffolded all modules.
- **Tool used:** Claude Code (claude-sonnet-4-6) for boilerplate scaffolding and architecture sounding-board.

### Hour 2 (1:00–2:00) — Core Modules

- Implemented `tutor/curriculum_loader.py`: load, filter, sample_diagnostic_probes, stem(), tts_path().
- Implemented `tutor/adaptive.py`: BKTSkillState with standard posterior update, EloSkillState, LearnerState with select_next_item() (zone of proximal development targeting), dyscalculia_warning() (3+ session plateau detection).
- Implemented `tutor/lang_detect.py`: lexicon-based + n-gram hybrid, 'mix' detection, reply_lang helper.
- Implemented `tutor/asr_adapt.py`: Whisper-tiny wrapper, pitch normalisation for child voices (-4.5 semitones), number-word extraction (EN/FR/KIN), silence detection.
- **Tool used:** Claude Code for BKT math review and soundfile integration.

### Hour 3 (2:00–3:00) — Pipeline, UI, Data Generator

- Implemented `tutor/visual_grounding.py`: BlobCounter baseline + OWLViT-tiny optional backend, render_counting_stimulus().
- Implemented `tutor/progress_store.py`: AES-256-GCM encrypted SQLite, learner switching (PIN hash), session management, weekly_report(), dp_sync_payload() with Gaussian mechanism (ε = 1.0).
- Implemented `tutor/model_loader.py`: GGUF llama-cpp-python wrapper + template feedback fallback.
- Implemented `demo.py` (Gradio): full session flow, first-90-second onboarding design, silence handling, multilingual feedback.
- Implemented `parent_report.py`: HTML/JSON/text output, icon-based non-literate-friendly design.
- Implemented `scripts/generate_curriculum.py`: 80-item curriculum generator (counting 1–15, addition/subtraction pairs, culturally grounded RWF word problems).
- Implemented `scripts/make_synthetic_child.py`: espeak-ng TTS + pitch-shift + MUSAN-style noise.
- **Tool used:** Claude Code for Gradio event wiring and HTML report layout.

### Hour 4 (3:00–4:00) — Evaluation, Documentation, Push

- Built `notebooks/kt_eval.ipynb`: 200 simulated learners × 40 responses, BKT vs Elo AUC evaluation, calibration curves, item-selection diversity.
- Wrote `footprint_report.md` with component-level size analysis.
- Wrote `README.md` (2-command setup, architecture diagram, product & business adaptation answers).
- Wrote `SIGNED.md`, `process_log.md`.
- Pushed all code to GitHub.
- **Tool used:** Claude Code for notebook cell structuring.

### Hour 5 (4:00–5:00) — Audit, Bug Fixes, Real Notebook Execution

- Full technical audit against 5/5 rubric; identified three gaps: fabricated AUC numbers in README, no real adapter weights committed, notebook not executed.
- Fixed `visual_grounding.py` blob counter binary inversion bug (`gray > thresh` → `gray < thresh`): BlobCounter now 100% accurate for n=1–10.
- Fixed `progress_store.py` dead code: removed `os.urandom(4).__len__()` (always returned 4), moved `import numpy as np` to module level.
- Fixed BKT smoke test: corrected wrong assertion (BKT always applies learning transition after wrong answers — mastery never decreases below initial).
- Fixed `render_counting_stimulus(0)` ZeroDivisionError: added early return for n=0, added regression test.
- Executed `notebooks/kt_eval.ipynb` with real kernel: BKT AUC = **0.5677**, Elo AUC = **0.5203** — replaced fabricated ~0.72 in README with real numbers and explanation.
- Updated README AUC table and added calibration curve explanation.
- **Tool used:** Claude Code for audit cross-referencing and nbformat execution.

### Hour 6 (5:00–6:00) — LoRA Training (CPU + GPU)

- Wrote `scripts/make_instruction_data.py`: generates 2,000 EN/FR/KIN instruction pairs (10 correct + 10 wrong templates per language × skill variations).
- Wrote `scripts/train_lora_mini.py`: distilgpt2 QLoRA proof-of-concept, runs on CPU in ~17 s. Fixed `rename_column` error and removed invalid `no_cuda`/`use_cpu` TrainingArguments.
- Ran `train_lora_mini.py` locally: committed real adapter weights to `tutor/adapters/distilgpt2-numeracy-lora/` — 405,504 trainable params, eval_loss = 5.576, 1.6 MB safetensors.
- Wrote `scripts/train_modal.py`: Modal.com T4 GPU training from VSCode terminal, no Colab required. Fixed three successive errors: `required=` kwarg not supported in older Modal, missing `rich` module, `eval_strategy` renamed to `evaluation_strategy` in transformers 4.40.
- Ran full production training on Modal Tesla T4 (15.6 GB VRAM): **3 epochs, train_loss = 0.4768, 793 s**. Downloaded adapter (8.6 MB) to `tutor/adapters/tinyllama-numeracy-lora/`.
- Updated `footprint_report.md`: live `du -sh tutor/` = 24 MB (✓ < 75 MB).
- **Tool used:** Claude Code for Modal API and PEFT version compatibility fixes.

### Hour 7 (6:00–7:00) — HuggingFace Publish, Interaction Logging, Roadmap

- Pulled tokenizer and config files from Modal volume (`modal volume get`) to complete the adapter directory.
- Pushed production TinyLlama adapter to HuggingFace Hub at `Nyingi101/math-tutor-tinyllama-lora` using write-scoped token.
- Created and pushed HuggingFace model card (README.md on the Hub) with YAML metadata (language tags en/fr/rw, license apache-2.0, base_model, pipeline_tag, training loss metric), usage examples in all three languages, LoRA config, training stats, and link back to GitHub repo.
- Added `log_interaction()` and `export_interactions_for_finetuning()` to `progress_store.py`: every prompt → child response → AI feedback triple is now stored AES-256-GCM encrypted on-device; one command exports pseudonymised JSONL for next training round.
- Added new `interactions` SQLite table to schema.
- Expanded README Next Steps section: 5 initiatives with Goal, Why It Matters, How, and Impact written for pitch deck use — school pilot, real data pipeline, GGUF quantisation, KIN ASR fine-tuning, teacher dashboard.
- **Tool used:** Claude Code for HuggingFace Hub API and SQLite schema extension.

---

## LLM & Tool Use Declaration

| Tool | Why used | Sample prompts |
|------|----------|----------------|
| **Claude Code (claude-sonnet-4-6)** | Architecture scaffolding, BKT math review, Gradio wiring, HTML layout | See below |

### Three Sample Prompts Actually Sent

**Prompt 1 (architecture):**
> "I'm building an offline AI math tutor for P1–P3 children in Rwanda. The constraint is ≤ 75 MB on-device and < 2.5 s latency on Colab CPU. Should I use BKT or DKT for knowledge tracing given the 4-hour build window? Walk me through the trade-offs."

**Prompt 2 (BKT update formula):**
> "Show me the standard BKT posterior update equation and a Python implementation of `p_known` given `p_learn`, `p_guess`, `p_slip`. I want the update to happen in one method call on a dataclass."

**Prompt 3 (Gradio state):**
> "In Gradio 4.x, how do I share mutable session state across multiple button click handlers without using global variables? Show me a pattern using gr.State() with a dict."

**Prompt 4 (Modal GPU training):**
> "I need to run TinyLlama QLoRA fine-tuning from my VSCode terminal without Colab. Write a Modal.com script that provisions a T4 GPU, uploads my instruction_data.jsonl, trains with 4-bit NF4 quantisation and LoRA rank=8, and downloads the adapter back locally. Keep it under $1 using Modal's free tier."

**Prompt 5 (real interaction data pipeline):**
> "The model was trained on synthetic data. Design a SQLite schema and two methods — one to log encrypted prompt/response/feedback triples on-device, one to export pseudonymised JSONL for retraining — that plug into the existing ProgressStore class without breaking the current API."

### One Prompt Discarded and Why

**Discarded:** "Generate all 80 curriculum items for me in one shot."

**Why discarded:** Handing curriculum generation entirely to the LLM would produce items without cultural grounding (RWF prices, Kinyarwanda contexts) and without the systematic difficulty calibration I needed for the BKT evaluation. I kept the generation logic explicit and code-driven so I could defend every item in the Live Defense.

---

## Hardest Decision

The single hardest decision was **how to handle the 75 MB footprint constraint while still having a meaningful language head**.

The spec asks for QLoRA fine-tuning + int4 quantisation of TinyLlama or Phi-3-mini. But even Q4_K_M TinyLlama is ~669 MB — nearly 9× the budget. Options considered:

1. **Ignore the constraint, ship the GGUF anyway** — fails the spec, easy to disqualify on.
2. **Use a tiny fine-tuned GPT-2-small** — fits (~28 MB at int4), but GPT-2 small is not a "language head" in the instruction-following sense the spec expects.
3. **Template feedback + documented GGUF upgrade path** — ship with 0 MB LLM, template responses, and a clear `model_loader.py` that hot-loads the GGUF from Hugging Face on first run. The *on-device* footprint stays ≤ 75 MB (model is not cached locally until first run, matching the spec's "total app footprint" framing). Document the full LoRA training recipe in the README.

I chose option 3. It is honest about the constraint, keeps latency well under 2.5 s, and demonstrates I understand the deployment reality (bandwidth-constrained first run vs. subsequent cached runs). The LoRA training recipe is fully documented even if the merged model is too large to bundle.

---

## Second Hard Decision — GPU Training Without Colab

The production LoRA spec requires TinyLlama fine-tuning on a GPU. The only option available during the build was Google Colab, which requires switching browser tabs, managing sessions, and re-uploading data on disconnect — not reproducible from the repo alone.

Instead I wrote `scripts/train_modal.py` using Modal.com serverless GPU. This was harder than it sounds: three successive API compatibility errors had to be debugged remotely (Modal `required=` kwarg, missing `rich` module in the container, `eval_strategy` renamed in transformers 4.40). Each required understanding the dependency pinning inside the Modal container image, not just the local environment.

The result is that **any judge or user can reproduce the full production training with two commands** (`modal token new` + `modal run scripts/train_modal.py`) at a cost of ~$0.88 — without Colab, without leaving the terminal, and with the adapter downloaded automatically to the correct local path. That reproducibility was worth the extra debugging time.

---

## Key Metrics Summary

| Metric | Value |
| ------ | ----- |
| Smoke tests passing | 19 / 19 ✓ |
| BKT AUC (200 learners × 40 responses) | 0.5677 |
| Elo AUC baseline | 0.5203 |
| BKT vs Elo delta | +0.0474 |
| Instruction pairs (EN/FR/KIN) | 2,000 |
| CPU adapter (distilgpt2) — trainable params | 405,504 (0.49%) |
| CPU adapter training time | ~17 s on CPU |
| Production adapter (TinyLlama) — trainable params | 2,252,800 (0.20%) |
| Production training time | 793 s on Tesla T4 |
| Production train loss | 0.4768 (converged) |
| `tutor/` footprint | 24 MB (target ≤ 75 MB ✓) |
| Latency (template mode) | < 50 ms |
| Supported languages | English · French · Kinyarwanda · code-switched |
