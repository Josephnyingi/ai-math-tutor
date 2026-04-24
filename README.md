# AI Math Tutor for Early Learners

**S2.T3.1 · AIMS KTT Hackathon · Tier 3**

An offline, adaptive AI math tutor for children aged 5–9 (P1–P3 numeracy).
Teaches counting, number sense, addition, subtraction, and word problems through
visuals, audio, and voice interaction. Works fully offline. Handles Kinyarwanda,
French, English, and code-switched input.

```
du -sh tutor/
< 2.0M   tutor/        (well within 75 MB target)
```

---

## 2-Command Setup (free Colab CPU)

```bash
pip install -r requirements.txt
python scripts/generate_curriculum.py && python demo.py
```

Open `http://localhost:7860` in your browser.

---

## Architecture

```
demo.py  (Gradio UI)
    │
    ├── tutor/curriculum_loader.py   — load / filter / sample items (JSON)
    ├── tutor/adaptive.py            — BKT + Elo knowledge tracing, item selection
    ├── tutor/asr_adapt.py           — Whisper-tiny + child pitch normalisation
    ├── tutor/lang_detect.py         — KIN/FR/EN/mix detection (lexicon + n-gram)
    ├── tutor/visual_grounding.py    — BlobCounter baseline + OWLViT-tiny optional
    ├── tutor/model_loader.py        — template feedback + GGUF upgrade path
    └── tutor/progress_store.py      — AES-256-GCM SQLite + ε-DP sync payload

scripts/
    ├── generate_curriculum.py       — reproducible 80-item curriculum generator
    └── make_synthetic_child.py      — TTS + pitch-shift + noise augmentation

parent_report.py                     — weekly HTML/JSON/text parent report
notebooks/kt_eval.ipynb              — BKT vs Elo AUC evaluation
```

---

## Technical Components

### 1 · On-Device Inference Pipeline
- **ASR**: `openai/whisper-tiny` (39 M params, CPU-only)
- **Child pitch normalisation**: input audio shifted –4.5 semitones before decoding
- **Latency**: stimulus → response → feedback < 2.5 s on Colab CPU
- **Silence handling**: 2 consecutive silent responses → gentle re-prompt in child's language

### 2 · Knowledge Tracing (BKT + Elo)

Bayesian Knowledge Tracing with per-skill state:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| p_learn | 0.20 | Probability of learning after each attempt |
| p_guess | 0.25 | P(correct \| not known) |
| p_slip | 0.10 | P(incorrect \| known) |
| p_known | 0.10 (prior) | Initial belief of mastery |

BKT posterior update:

```
P(known | correct) = P(known) × (1-p_slip) / [P(known)(1-p_slip) + (1-P(known))×p_guess]
P(known | wrong)   = P(known) × p_slip    / [P(known)×p_slip    + (1-P(known))×(1-p_guess)]
P(known_t+1)       = P(known_t | obs) + (1 - P(known_t | obs)) × p_learn
```

**Elo baseline**: per-skill rating updated via standard K=32 factor; item difficulty mapped to Elo range.

**AUC results** (200 simulated learners × 40 responses, 70/30 train/test split):

| Model | AUC |
|-------|-----|
| BKT | ~0.72 |
| Elo baseline | ~0.67 |

See `notebooks/kt_eval.ipynb` for full evaluation.

### 3 · Language Head (QLoRA / LoRA)

**Training recipe** (documented; model hosted on Hugging Face):
- Base: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Dataset: numeracy feedback instruction pairs (EN/FR/KIN), ~2 000 examples
- LoRA rank=8, alpha=16, target modules: q_proj, v_proj
- Quantisation: Q4_K_M GGUF via llama.cpp
- Merged model: [huggingface.co/Josephnyingi/math-tutor-tinyllama-q4](https://huggingface.co/Josephnyingi/math-tutor-tinyllama-q4) *(upload in progress)*

**On-device behaviour**: `model_loader.py` uses template responses (0 MB, < 50 ms) by default.
To load the GGUF: `tutor.model_loader.set_model_path('path/to/model.gguf')`.

### 4 · Multilingual + Code-Switch Detection

```python
from tutor.lang_detect import detect, reply_lang

dominant, scores = detect("neuf gatatu")   # → ("mix", {"fr": 0.48, "kin": 0.45, "en": 0.07})
lang = reply_lang(dominant, fallback="en") # → "fr"  (highest single score)
```

For mixed responses: reply in dominant language, preserve number words from second language in feedback.

### 5 · Visual Grounding

Every `counting` skill item renders a stimulus image (coloured circles / emoji objects).
The model counts objects via:
- **BlobCounter** (default): connected-component analysis, no dependencies
- **OWLViT-tiny** (optional): zero-shot detection via `transformers` — `pip install transformers torch`

```python
from tutor.visual_grounding import render_counting_stimulus, count_objects
img = render_counting_stimulus(5)       # renders 5 blue circles
n, backend = count_objects(img, "goats") # → (5, "blob")
```

### 6 · Progress Store + DP Sync

- **Storage**: AES-256-GCM encrypted SQLite (`tutor_progress.db`)
- **Learner switching**: PIN hash (SHA-256); PIN-free tap also supported
- **Weekly report**: see `parent_report.py`
- **Differential privacy**: Gaussian mechanism, ε = 1.0 per learner per week

```
Privacy budget:  ε = 1.0,  δ = 1e-5  per learner per week
Sensitivity:     1.0 (per-skill accuracy ∈ [0,1])
Noise σ:         √(2 ln(1.25/δ)) × 1.0 / 1.0 ≈ 4.65
```

Only noisy cohort averages leave the device. No individual response records are ever synced.

### 7 · Footprint ≤ 75 MB

```
tutor/              < 2 MB   (Python source + curriculum JSON)
TTS cache           excluded (Coqui/Piper cache per spec)
Whisper-tiny        cached in ~/.cache/whisper/ (NOT in tutor/)
Progress DB         < 1 MB   (SQLite, local only)
─────────────────────────────
Total on-device     < 3 MB   ✓
```

---

## Curriculum

80-item curriculum across 5 sub-skills and 4 age bands (5–6, 6–7, 7–8, 8–9).
Difficulty scale 1–10. All items have EN, FR, KIN stems.

| Skill | Items | Difficulty range |
|-------|-------|-----------------|
| counting | 18 | 1–3 |
| number_sense | 16 | 3–9 |
| addition | 23 | 3–9 |
| subtraction | 23 | 3–9 |
| word_problem | 10 | 6–9 |

Regenerate from seed in < 30 s:

```bash
python scripts/generate_curriculum.py \
  --seed data/T3.1_Math_Tutor/curriculum_seed.json \
  --out  data/T3.1_Math_Tutor/curriculum_full.json
```

---

## Product & Business Adaptation

### First 90 Seconds (6-year-old Kinyarwanda speaker, first launch)

| Time | What happens |
|------|-------------|
| 0 s | App opens. Warm voice plays: *"Muraho! Nzina uyu mukino wa imibare."* (KIN) → *"Hello! Let's play math!"* (EN). Big colourful emoji fill the screen — no text. |
| 3 s | A single large ▶ button with a star icon pulses. No text required to tap it. |
| 10 s | If child hasn't tapped: voice repeats + animated hand-pointer bounces toward the button. No timeout penalty. |
| 15 s | First diagnostic probe appears: *"Pome zingahe?"* (KIN) with a rendered image of 3 apples. Three large tap-buttons: 2 · 3 · 4. |
| 30 s | Child taps or speaks. Immediate audio feedback. Happy sound if correct. Gentle "try again" sound + highlight if wrong — no shame. |
| 60 s | Five diagnostic probes complete. BKT initialised. Adaptive session begins. |
| 90 s | Child is in their personalised learning flow at the right difficulty for their skill level. |

**Silence for 10 seconds**: gentle audio re-prompt in Kinyarwanda plays. Second silence → switch to tap-only mode automatically. Third silence → large blinking hand-pointer bounces toward the tap buttons. No error message, no penalty.

### Shared Tablet (3 children, community centre)

- **Learner switching**: home screen shows avatar icons (no names — privacy). Each child taps their own icon. Optional 4-digit PIN (shown as coloured dots, not numbers).
- **Privacy**: each learner's data is encrypted under a separate AES key derived from their PIN. No learner can read another's data without their PIN.
- **Graceful reboot**: `LearnerState` is serialised to encrypted SQLite after every response. On reboot, the app recovers the latest state automatically — the child resumes exactly where they left off.
- **Offline-first**: zero network calls at inference. No cloud dependency. The tablet works with no WiFi indefinitely.
- **Power failure**: SQLite writes are committed per-response (not per-session), so a hard reboot loses at most one response.

### Non-Literate Parent Report

The weekly report (`python parent_report.py LEARNER_ID --format html --lang kin`) produces:

1. **Three big icons** at the top:
   - ⬆️ / ➡️ / ⬇️ — overall trend (improvement vs flat vs decline)
   - ⭐ — best skill this week (icon + Kinyarwanda label)
   - ⚠️ or ✅ — skill that needs attention (or all good)

2. **Five skill bars** — colour bars only (green / amber / red), no numbers. Visible at a glance.

3. **Session count** — shown as dots (●●● = 3 sessions), not a number.

4. **QR code** → a 20-second TTS audio file in Kinyarwanda: *"Amani yakoze neza cyane iyi cyumweru. Gutunga ni byiza. Imibare y'indangagaciro ikenera akazi kenshi."* ("Amani did very well this week. Counting is good. Number sense needs more practice.")

A non-literate parent can understand the full report in under 60 seconds with no reading required.

---

## Dyscalculia Early-Warning (Stretch Goal)

If a learner's BKT mastery plateaus for 3+ consecutive sessions despite the adaptive engine dropping to difficulty ≤ 3, the system:
1. Adds a soft ⚠️ flag to the parent report ("Talk to a teacher").
2. Logs the flag in the SQLite DB for teacher review.
3. Does NOT label the child as "dyscalculic" — only surfaces a gentle signal.

---

## Running the Parent Report

```bash
# HTML (for printing / WhatsApp sharing)
python parent_report.py amani --format html --lang kin --out report_amani.html

# JSON (for teacher dashboard integration)
python parent_report.py amani --format json

# Plain text (terminal)
python parent_report.py amani --format text
```

---

## Model Hosting

| Artefact | Location |
|----------|----------|
| Code | https://github.com/Josephnyingi/ai-math-tutor |
| Curriculum generator | `scripts/generate_curriculum.py` (in repo) |
| TinyLlama Q4 LoRA *(optional)* | https://huggingface.co/Josephnyingi/math-tutor-tinyllama-q4 |
| process_log.md | repo root |
| SIGNED.md | repo root |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Whisper-tiny accuracy degrades for Kinyarwanda | Pitch normalisation + KIN lexicon post-processing; tap-fallback always available |
| Tablet has < 1 GB RAM | Whisper-tiny unloaded between sessions; only one skill model in memory at a time |
| Child speaks very quietly | Silence threshold tuned conservatively (0.01 RMS); auto-retry |
| Intermittent power cuts data | Per-response SQLite commits; full recovery on reboot |
| Privacy breach (stolen tablet) | AES-256-GCM encryption; data useless without PIN |
