"""
demo.py — Child-facing Gradio demo for the AI Math Tutor.

Run:
    pip install -r requirements.txt
    python3 scripts/generate_curriculum.py   # one-time
    python3 demo.py                          # opens http://localhost:7860
"""
from __future__ import annotations

import time
from pathlib import Path

import gradio as gr
import numpy as np

from tutor import curriculum_loader as cl
from tutor.adaptive import LearnerState
from tutor.lang_detect import detect as lang_detect, reply_lang
from tutor.asr_adapt import transcribe, extract_integer, is_silence
from tutor.visual_grounding import render_counting_stimulus
from tutor.model_loader import generate_feedback
from tutor.progress_store import ProgressStore

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/T3.1_Math_Tutor")
DB_PATH = Path("tutor_progress.db")
CURRICULUM_PATH = DATA_DIR / "curriculum_full.json"
if not CURRICULUM_PATH.exists():
    CURRICULUM_PATH = DATA_DIR / "curriculum_seed.json"

ALL_ITEMS = cl.load(CURRICULUM_PATH)
STORE = ProgressStore(DB_PATH)

# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def new_session(learner_id: str, lang: str = "en", age: int = 7) -> dict:
    saved = STORE.load_latest_state(learner_id)
    if saved:
        state = LearnerState.from_dict(saved)
        state.age = age
    else:
        state = LearnerState(learner_id=learner_id, lang=lang, age=age)
        STORE.add_learner(learner_id, display_name=learner_id)
    state.lang = lang
    cfg = state.age_config
    probes = cl.sample_diagnostic_probes(
        ALL_ITEMS, n_per_skill=1,
        diff_min=cfg["diff_min"], diff_max=cfg["diff_max"],
    )
    return {
        "learner_id": learner_id,
        "lang": lang,
        "age": age,
        "state": state,
        "queue": probes,
        "current_item": None,
        "session_id": STORE.start_session(learner_id, state.to_dict(), lang),
        "phase": "diagnostic",
        "silence_count": 0,
        "total_correct": 0,
        "total_answered": 0,
    }


def get_next_item(sess: dict) -> dict:
    state: LearnerState = sess["state"]
    if sess["queue"]:
        item = sess["queue"].pop(0)
        if not sess["queue"]:
            sess["phase"] = "adaptive"
    else:
        item = state.select_next_item(ALL_ITEMS, use_bkt=True)
    sess["current_item"] = item
    return sess


# ---------------------------------------------------------------------------
# Core response processor
# ---------------------------------------------------------------------------

def process_response(
    sess: dict,
    audio_data: tuple | None,
    tap_answer: str,
) -> tuple[dict, str, bool, np.ndarray | None, str]:
    """Returns (sess, feedback_text, is_correct, image, debug)."""
    item = sess.get("current_item")
    if item is None:
        return sess, "Let's start!", False, None, ""

    state: LearnerState = sess["state"]
    lang = sess["lang"]
    t0 = time.time()

    child_text = ""

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
            return sess, _silence_prompt(lang), False, _item_image(item), ""

        sess["silence_count"] = 0
        child_text, detected_lang, _ = transcribe(audio_f32, lang_hint=lang)
        if detected_lang == "mix":
            lang = reply_lang(detected_lang, fallback=lang)
        elif detected_lang in ("en", "fr", "kin", "sw"):
            lang = detected_lang
        sess["lang"] = lang

    elif tap_answer.strip():
        child_text = tap_answer.strip()

    child_int = extract_integer(child_text)
    is_correct = child_int is not None and child_int == item["answer_int"]

    image_arr = _item_image(item)

    state.record_response(item, is_correct)
    latency_ms = int((time.time() - t0) * 1000)
    STORE.log_response(
        sess["learner_id"], sess["session_id"],
        item["id"], item["skill"],
        item.get("difficulty", 5), is_correct, latency_ms,
    )

    feedback = generate_feedback(is_correct, item["answer_int"], lang, child_text)

    warnings = state.dyscalculia_warning()
    if warnings:
        feedback += f"\n\n[Parent: {', '.join(warnings)} skill(s) may need attention]"

    sess["total_answered"] = sess.get("total_answered", 0) + 1
    if is_correct:
        sess["total_correct"] = sess.get("total_correct", 0) + 1

    STORE.end_session(sess["session_id"], state.to_dict())
    sess = get_next_item(sess)

    debug = f"item={item['id']} correct={is_correct} lang={lang} latency={latency_ms}ms"
    return sess, feedback, is_correct, image_arr, debug


def _item_image(item: dict) -> np.ndarray | None:
    if item["skill"] == "counting" and item.get("answer_int"):
        label = item.get("visual", "●").split("_")[0]
        return render_counting_stimulus(item["answer_int"], label=label)
    return None


def _silence_prompt(lang: str) -> str:
    msgs = {
        "en": "I didn't hear you — tap a number or try again!",
        "fr": "Je ne t'ai pas entendu — touche un chiffre !",
        "kin": "Sinumvise — kanda umubare!",
        "sw": "Sikukusikia — gonga nambari!",
    }
    return msgs.get(lang, msgs["en"])


# ---------------------------------------------------------------------------
# HTML renderers
# ---------------------------------------------------------------------------

def _question_html(text: str) -> str:
    return (
        '<div style="font-size:1.7em; font-weight:800; text-align:center; '
        'color:#1a3a8f; padding:22px 16px; background:linear-gradient(135deg,#e8f0fe,#f3e8ff); '
        'border-radius:20px; min-height:90px; display:flex; align-items:center; '
        f'justify-content:center; line-height:1.3">{text}</div>'
    )


def _feedback_html(text: str, is_correct: bool) -> str:
    if not text:
        return ""
    if is_correct:
        return (
            '<div style="background:#d4edda; border:3px solid #28a745; border-radius:20px; '
            'padding:18px; text-align:center; font-size:1.5em; font-weight:800; color:#155724; '
            'animation:pop .25s ease" class="fb-panel">'
            f'✅ {text}</div>'
            '<style>@keyframes pop{{0%{{transform:scale(.85)}}100%{{transform:scale(1)}}}}'
            '.fb-panel{{animation:pop .25s ease}}</style>'
        )
    return (
        '<div style="background:#fff3cd; border:3px solid #ffc107; border-radius:20px; '
        'padding:18px; text-align:center; font-size:1.5em; font-weight:800; color:#7d5a00">'
        f'💭 {text}</div>'
    )


def _progress_html(correct: int, answered: int) -> str:
    stars = min(correct, 10)
    bar = "⭐" * stars + "☆" * (10 - stars)
    label = f"{correct}/{answered} correct" if answered else "Answer to earn stars!"
    return (
        f'<div style="text-align:center; padding:8px 0">'
        f'<div style="font-size:1.6em; letter-spacing:3px">{bar}</div>'
        f'<div style="font-size:0.85em; color:#666; margin-top:2px">{label}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

# Number button colours (0–10)
_NUM_COLORS = [
    "#6c757d",  # 0 grey
    "#2ecc71",  # 1 green
    "#1abc9c",  # 2 teal
    "#3498db",  # 3 blue
    "#9b59b6",  # 4 purple
    "#e91e63",  # 5 pink
    "#f39c12",  # 6 amber
    "#e74c3c",  # 7 red
    "#1e90ff",  # 8 dodger blue
    "#8e44ad",  # 9 violet
    "#2c3e50",  # 10 dark
]

_NUM_CSS = "\n".join(
    f'.nb{i} button {{ background:{c} !important; color:white !important; '
    f'font-size:1.7em !important; font-weight:900 !important; '
    f'border-radius:14px !important; height:68px !important; '
    f'transition:transform .1s !important; border:none !important; }}\n'
    f'.nb{i} button:hover {{ transform:scale(1.07) !important; }}\n'
    f'.nb{i} button:active {{ transform:scale(0.93) !important; }}'
    for i, c in enumerate(_NUM_COLORS)
)

_GLOBAL_CSS = _NUM_CSS + """
.start-btn button {
    font-size:1.3em !important; font-weight:800 !important;
    border-radius:16px !important; height:60px !important;
}
"""


def build_ui():
    theme = gr.themes.Soft(
        primary_hue="blue",
        font=[gr.themes.GoogleFont("Nunito"), "sans-serif"],
    )

    with gr.Blocks(title="🌟 Math Adventure", css=_GLOBAL_CSS) as demo:
        sess_state = gr.State(None)

        # ── Header ──────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; padding:18px 12px 14px;
                    background:linear-gradient(135deg,#1a56db,#9333ea);
                    border-radius:20px; margin-bottom:14px">
          <div style="font-size:3.2em; line-height:1.1">🦁</div>
          <h1 style="color:white; font-size:2.1em; margin:4px 0 2px; font-weight:900">
            Math Adventure!
          </h1>
          <p style="color:rgba(255,255,255,0.88); font-size:0.95em; margin:0">
            Akanyamaswa k'Imibare &nbsp;·&nbsp; Aventure Maths &nbsp;·&nbsp; Hisabu ya Kusisimua
          </p>
        </div>

        <!-- For Parents & Teachers collapsible panel -->
        <details style="margin-bottom:14px; border:2px solid #d1d5db; border-radius:14px;
                        background:#f9fafb; padding:0; overflow:hidden">
          <summary style="cursor:pointer; padding:12px 18px; font-weight:800; font-size:1em;
                          color:#374151; list-style:none; display:flex; align-items:center; gap:8px;
                          user-select:none">
            👨‍👩‍👧 <span>For Parents &amp; Teachers</span>
            <span style="margin-left:auto; font-size:0.8em; color:#6b7280">tap to expand</span>
          </summary>
          <div style="padding:14px 18px 16px; border-top:1px solid #e5e7eb; font-size:0.95em; color:#374151; line-height:1.7">
            <p style="margin:0 0 10px"><strong>How to set up a session:</strong></p>
            <ol style="margin:0 0 12px; padding-left:20px">
              <li>Enter the child's <strong>name</strong> so progress is saved across sessions.</li>
              <li>Select their <strong>age (5–9)</strong> — this sets the difficulty level automatically. Younger children see easier questions; older children get harder ones.</li>
              <li>Pick the <strong>language</strong> the child is most comfortable with.</li>
              <li>Press <strong>START</strong> and hand the device to the child.</li>
            </ol>
            <p style="margin:0 0 10px"><strong>How the child answers:</strong></p>
            <ul style="margin:0 0 12px; padding-left:20px">
              <li>👇 <strong>Tap a coloured number button</strong> — no reading or typing needed.</li>
              <li>🎤 Or <strong>speak the answer</strong> into the microphone (optional).</li>
              <li>⭐ Stars fill up as the child gets answers right.</li>
            </ul>
            <p style="margin:0 0 6px"><strong>Viewing the child's progress report:</strong></p>
            <code style="background:#e5e7eb; padding:4px 8px; border-radius:6px; font-size:0.9em">
              python3 parent_report.py &lt;name&gt; --lang kin
            </code>
            <p style="margin:8px 0 0; color:#6b7280; font-size:0.88em">
              Generates a one-page visual report (colour bars, skill badges, trend arrows)
              designed for non-literate parents — no numbers, just icons and colours.
            </p>
          </div>
        </details>
        """)

        with gr.Row(equal_height=False):

            # ── Setup panel (left) ───────────────────────────────────────
            with gr.Column(scale=1, min_width=210):
                learner_id_box = gr.Textbox(
                    label="👤 Name / Izina / Nom / Jina",
                    placeholder="e.g. Amani",
                    max_lines=1,
                )
                age_radio = gr.Radio(
                    choices=[("5 yrs 🐣", 5), ("6 yrs 🐥", 6), ("7 yrs 🌱", 7),
                             ("8 yrs 🌟", 8), ("9 yrs 🚀", 9)],
                    value=7,
                    label="🎂 Age / Imyaka / Âge / Umri",
                )
                lang_radio = gr.Radio(
                    choices=[
                        ("🇷🇼 Kinyarwanda", "kin"),
                        ("🇹🇿 Kiswahili", "sw"),
                        ("🇫🇷 Français", "fr"),
                        ("🇬🇧 English", "en"),
                    ],
                    value="kin",
                    label="🌍 Language / Lugha / Langue / Ururimi",
                )
                start_btn = gr.Button(
                    "▶  START / TANGIRA",
                    variant="primary",
                    size="lg",
                    elem_classes=["start-btn"],
                )

            # ── Game panel (right) ───────────────────────────────────────
            with gr.Column(scale=3):

                progress_html = gr.HTML(_progress_html(0, 0))

                question_html = gr.HTML(
                    _question_html("Press START to begin! 👆")
                )

                item_image = gr.Image(label="", height=270, show_label=False)

                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="🎤 Speak / Vuga / Parle / Sema  (optional)",
                )

                gr.HTML(
                    '<p style="text-align:center; font-weight:800; font-size:1.1em; '
                    'color:#444; margin:10px 0 4px">👇 Tap your answer:</p>'
                )

                # Number pad row 1: 0–5
                with gr.Row():
                    num_btns_r1 = [
                        gr.Button(str(n), elem_classes=[f"nb{n}"])
                        for n in range(6)
                    ]
                # Number pad row 2: 6–10
                with gr.Row():
                    num_btns_r2 = [
                        gr.Button(str(n), elem_classes=[f"nb{n}"])
                        for n in range(6, 11)
                    ]

                feedback_html = gr.HTML("")

                debug_box = gr.Textbox(
                    label="Debug", visible=False, interactive=False
                )

        # ── Shared output list ───────────────────────────────────────────
        OUTPUTS = [
            sess_state,
            question_html,
            item_image,
            audio_input,
            progress_html,
            feedback_html,
            debug_box,
        ]

        # ── on_start ─────────────────────────────────────────────────────
        def on_start(learner_id, age, lang):
            if not learner_id.strip():
                learner_id = "learner_1"
            sess = new_session(learner_id.strip(), lang, age=int(age))
            sess = get_next_item(sess)
            item = sess["current_item"]
            if item:
                q = cl.stem(item, lang)
                img = _item_image(item)
            else:
                q, img = "No items available", None
            return (
                sess,
                _question_html(q),
                img,
                None,
                _progress_html(0, 0),
                "",
                "",
            )

        start_btn.click(
            on_start,
            inputs=[learner_id_box, age_radio, lang_radio],
            outputs=OUTPUTS,
        )

        # ── on_submit (shared by audio + number pad) ──────────────────────
        def on_submit(sess, audio, tap_answer):
            if sess is None:
                return (
                    None,
                    _question_html("Press START first! 👆"),
                    None, None,
                    _progress_html(0, 0),
                    "",
                    "no session",
                )
            sess, feedback, is_correct, img, debug = process_response(
                sess, audio, tap_answer
            )
            item = sess.get("current_item")
            q = cl.stem(item, sess["lang"]) if item else "🎉 All done! Great work!"
            return (
                sess,
                _question_html(q),
                img,
                None,
                _progress_html(sess.get("total_correct", 0), sess.get("total_answered", 0)),
                _feedback_html(feedback, is_correct),
                debug,
            )

        # Number pad buttons auto-submit with their value
        all_num_btns = num_btns_r1 + num_btns_r2
        for btn in all_num_btns:
            num_val = btn.value  # "0" … "10"
            btn.click(
                fn=lambda s, a, n=num_val: on_submit(s, a, n),
                inputs=[sess_state, audio_input],
                outputs=OUTPUTS,
            )

        # Audio submit (after recording)
        audio_input.change(
            fn=lambda s, a: on_submit(s, a, ""),
            inputs=[sess_state, audio_input],
            outputs=OUTPUTS,
        )

    return demo, theme


if __name__ == "__main__":
    app, theme = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=theme)
