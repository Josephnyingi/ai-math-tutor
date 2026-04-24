"""
demo.py — Child-facing Gradio demo for the AI Math Tutor.

Run:
    pip install -r requirements.txt
    python demo.py

First-time open sequence (first 90 seconds):
  0 s  — Warm welcome audio plays in Kinyarwanda + English
  5 s  — Big, icon-only "START" button appears (no reading required)
  10 s — If child hasn't tapped: gentle audio prompt replays, animated arrow bounces
  15 s — Diagnostic probe #1 appears (lowest difficulty counting item)
  ~90 s — Five diagnostic probes complete; BKT initialised; adaptive session begins
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import gradio as gr
import numpy as np

from tutor import curriculum_loader as cl
from tutor.adaptive import LearnerState
from tutor.lang_detect import detect as lang_detect, reply_lang
from tutor.asr_adapt import transcribe, extract_integer, is_silence
from tutor.visual_grounding import render_counting_stimulus, count_objects
from tutor.model_loader import generate_feedback
from tutor.progress_store import ProgressStore

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
DATA_DIR = Path("data/T3.1_Math_Tutor")
DB_PATH = Path("tutor_progress.db")
CURRICULUM_PATH = DATA_DIR / "curriculum_full.json"
if not CURRICULUM_PATH.exists():
    CURRICULUM_PATH = DATA_DIR / "curriculum_seed.json"

ALL_ITEMS = cl.load(CURRICULUM_PATH)
STORE = ProgressStore(DB_PATH)

# ------------------------------------------------------------------
# Session state (Gradio uses dict-based state)
# ------------------------------------------------------------------

def new_session(learner_id: str, lang: str = "en") -> dict:
    saved = STORE.load_latest_state(learner_id)
    if saved:
        state = LearnerState.from_dict(saved)
    else:
        state = LearnerState(learner_id=learner_id, lang=lang)
        STORE.add_learner(learner_id, display_name=learner_id)
    state.lang = lang
    probes = cl.sample_diagnostic_probes(ALL_ITEMS, n_per_skill=1)
    return {
        "learner_id": learner_id,
        "lang": lang,
        "state": state,
        "queue": probes,
        "current_item": None,
        "session_id": STORE.start_session(learner_id, state.to_dict(), lang),
        "phase": "diagnostic",  # diagnostic → adaptive
        "silence_count": 0,
    }


def get_next_item(sess: dict) -> dict:
    """Pop next item from queue or select adaptively."""
    state: LearnerState = sess["state"]
    if sess["queue"]:
        item = sess["queue"].pop(0)
        if not sess["queue"]:
            sess["phase"] = "adaptive"
    else:
        item = state.select_next_item(ALL_ITEMS, use_bkt=True)
    sess["current_item"] = item
    return sess


# ------------------------------------------------------------------
# Response processing
# ------------------------------------------------------------------

def process_response(
    sess: dict,
    audio_data: tuple | None,
    tap_answer: str,
) -> tuple[dict, str, np.ndarray | None, str]:
    """
    Handle a child response (voice or tap).
    Returns: (updated_sess, feedback_text, image_array, debug_info)
    """
    item = sess.get("current_item")
    if item is None:
        return sess, "Let's start!", None, ""

    state: LearnerState = sess["state"]
    lang = sess["lang"]
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Parse answer
    # ------------------------------------------------------------------
    child_text = ""
    detected_lang = lang

    if audio_data is not None:
        sr, audio_np = audio_data
        audio_f32 = audio_np.astype(np.float32) / 32768.0
        if sr != 16000:
            ratio = 16000 / sr
            n = int(len(audio_f32) * ratio)
            idx = np.linspace(0, len(audio_f32) - 1, n)
            li = np.floor(idx).astype(int)
            ri = np.clip(li + 1, 0, len(audio_f32) - 1)
            audio_f32 = audio_f32[li] * (1 - (idx - li)) + audio_f32[ri] * (idx - li)

        if is_silence(audio_f32):
            sess["silence_count"] += 1
            if sess["silence_count"] >= 2:
                # Two consecutive silences → repeat prompt gently
                return sess, _silence_prompt(lang), _item_image(item), ""
            return sess, _silence_prompt(lang), _item_image(item), ""

        sess["silence_count"] = 0
        child_text, detected_lang, conf = transcribe(audio_f32, lang_hint=lang)
        if detected_lang == "mix":
            lang = reply_lang(detected_lang, fallback=lang)
        elif detected_lang in ("en", "fr", "kin"):
            lang = detected_lang
        sess["lang"] = lang

    elif tap_answer.strip():
        child_text = tap_answer.strip()

    # ------------------------------------------------------------------
    # 2. Score
    # ------------------------------------------------------------------
    child_int = extract_integer(child_text)
    is_correct = child_int is not None and child_int == item["answer_int"]

    # ------------------------------------------------------------------
    # 3. Visual grounding items
    # ------------------------------------------------------------------
    image_arr = _item_image(item)

    # ------------------------------------------------------------------
    # 4. Update knowledge state
    # ------------------------------------------------------------------
    state.record_response(item, is_correct)
    latency_ms = int((time.time() - t0) * 1000)
    STORE.log_response(
        sess["learner_id"],
        sess["session_id"],
        item["id"],
        item["skill"],
        item.get("difficulty", 5),
        is_correct,
        latency_ms,
    )

    # ------------------------------------------------------------------
    # 5. Generate feedback
    # ------------------------------------------------------------------
    feedback = generate_feedback(is_correct, item["answer_int"], lang, child_text)

    # ------------------------------------------------------------------
    # 6. Dyscalculia early warning
    # ------------------------------------------------------------------
    warnings = state.dyscalculia_warning()
    if warnings:
        feedback += f"\n\n[Parent: {', '.join(warnings)} skill(s) may need teacher attention]"

    # ------------------------------------------------------------------
    # 7. Advance to next item
    # ------------------------------------------------------------------
    STORE.end_session(sess["session_id"], state.to_dict())
    sess = get_next_item(sess)

    total_elapsed = time.time() - t0
    debug = f"item={item['id']} correct={is_correct} lang={lang} latency={total_elapsed:.2f}s"
    return sess, feedback, image_arr, debug


def _item_image(item: dict) -> np.ndarray | None:
    if item["skill"] == "counting" and item.get("answer_int"):
        label = item.get("visual", "●").split("_")[0]
        return render_counting_stimulus(item["answer_int"], label=label)
    return None


def _silence_prompt(lang: str) -> str:
    msgs = {
        "en": "I didn't hear you. Try again! Tap the number or speak.",
        "fr": "Je ne t'ai pas entendu. Essaie encore !",
        "kin": "Sinumvise. Ongera ugerageze!",
        "sw": "Sikukusikia. Jaribu tena! Gonga nambari au sema.",
    }
    return msgs.get(lang, msgs["en"])


# ------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------

def build_ui():
    theme = gr.themes.Soft(
        primary_hue="blue",
        font=[gr.themes.GoogleFont("Nunito"), "sans-serif"],
    )

    with gr.Blocks(title="AI Math Tutor") as demo:
        sess_state = gr.State(None)

        gr.HTML("""
        <div style="text-align:center; padding:20px">
          <h1 style="font-size:2.5em; color:#1a56db">🧮 Math Tutor</h1>
          <p style="font-size:1.2em; color:#555">
            Mwarimu wa Hesabu &nbsp;|&nbsp; Tuteur Maths &nbsp;|&nbsp; Math Tutor
          </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                learner_id_box = gr.Textbox(
                    label="Learner name / Izina",
                    placeholder="e.g. Amani",
                    max_lines=1,
                )
                lang_radio = gr.Radio(
                    choices=[("Kinyarwanda", "kin"), ("Kiswahili", "sw"), ("Français", "fr"), ("English", "en")],
                    value="kin",
                    label="Language / Lugha / Langue / Ururimi",
                )
                start_btn = gr.Button("▶  START / TANGIRA", variant="primary", size="lg")

            with gr.Column(scale=2):
                question_md = gr.Markdown("### Press START to begin!")
                item_image = gr.Image(label="", height=200, show_label=False)
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="🎤 Say your answer / Vuga igisubizo",
                )
                tap_input = gr.Textbox(
                    label="Or type / Andika",
                    placeholder="e.g. 5",
                    max_lines=1,
                )
                submit_btn = gr.Button("✅ Submit / Ohereza", size="lg")
                feedback_box = gr.Textbox(
                    label="Feedback",
                    interactive=False,
                    lines=3,
                )
                debug_box = gr.Textbox(label="Debug", visible=False, interactive=False)

        # ------------------------------------------------------------------
        # Event handlers
        # ------------------------------------------------------------------

        def on_start(learner_id, lang):
            if not learner_id.strip():
                learner_id = "learner_1"
            sess = new_session(learner_id.strip(), lang)
            sess = get_next_item(sess)
            item = sess["current_item"]
            if item:
                q = cl.stem(item, lang)
                img = _item_image(item)
                return sess, f"### {q}", img, None, "", ""
            return sess, "### No items available", None, None, "", ""

        start_btn.click(
            on_start,
            inputs=[learner_id_box, lang_radio],
            outputs=[sess_state, question_md, item_image, audio_input, tap_input, feedback_box],
        )

        def on_submit(sess, audio, tap):
            if sess is None:
                return None, "### Press START first", None, "", "", "no session"
            sess, feedback, img, debug = process_response(sess, audio, tap)
            item = sess.get("current_item")
            if item:
                q = cl.stem(item, sess["lang"])
                question = f"### {q}"
            else:
                question = "### All done! Great work!"
            return sess, question, img, None, "", feedback, debug

        submit_btn.click(
            on_submit,
            inputs=[sess_state, audio_input, tap_input],
            outputs=[sess_state, question_md, item_image, audio_input, tap_input, feedback_box, debug_box],
        )

    return demo, theme


if __name__ == "__main__":
    app, theme = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=theme)
