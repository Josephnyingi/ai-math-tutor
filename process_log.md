# Process Log — S2.T3.1 AI Math Tutor

**Candidate:** Joseph Nyingi  
**Date:** 2026-04-24  
**Total time:** ~4 hours  

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
