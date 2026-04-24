"""
smoke_test.py — Verify all tutor modules work without heavy deps (Whisper, GGUF).

Run before Live Defense:
    python3 scripts/smoke_test.py

All tests must print PASS. Zero network calls. Runs in < 10 s.
"""
import sys
import os
import tempfile
import random
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
random.seed(42)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
errors = []


def test(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
    except Exception as e:
        print(f"  {FAIL}  {name}: {e}")
        errors.append((name, traceback.format_exc()))


# ------------------------------------------------------------------
print("\n=== curriculum_loader ===")

def t_load():
    from tutor import curriculum_loader as cl
    items = cl.load("data/T3.1_Math_Tutor/curriculum_full.json")
    assert len(items) >= 60, f"Expected ≥60 items, got {len(items)}"
    assert all("skill" in it and "answer_int" in it for it in items)

def t_filter():
    from tutor import curriculum_loader as cl
    items = cl.load("data/T3.1_Math_Tutor/curriculum_full.json")
    easy = cl.filter_items(items, skill="counting", max_diff=3)
    assert all(it["difficulty"] <= 3 for it in easy)

def t_probes():
    from tutor import curriculum_loader as cl
    items = cl.load("data/T3.1_Math_Tutor/curriculum_full.json")
    probes = cl.sample_diagnostic_probes(items)
    skills_covered = {p["skill"] for p in probes}
    assert len(skills_covered) == 5, f"Probes must cover all 5 skills, got {skills_covered}"

def t_stem():
    from tutor import curriculum_loader as cl
    items = cl.load("data/T3.1_Math_Tutor/curriculum_full.json")
    item = items[0]
    en = cl.stem(item, "en")
    assert isinstance(en, str) and len(en) > 0

test("load ≥60 items", t_load)
test("filter by skill/difficulty", t_filter)
test("diagnostic probes cover all 5 skills", t_probes)
test("stem() returns string", t_stem)


# ------------------------------------------------------------------
print("\n=== adaptive (BKT + Elo) ===")

def t_bkt_update():
    from tutor.adaptive import BKTSkillState
    # BKT applies the learning transition after EVERY response (correct or not).
    # A correct answer yields higher posterior mastery than a wrong one.
    s_correct = BKTSkillState()
    s_wrong = BKTSkillState()
    s_correct.update(True)
    s_wrong.update(False)
    assert s_correct.mastery > s_wrong.mastery, "Correct response must yield higher mastery"
    assert s_correct.mastery > BKTSkillState().p_known, "Correct must exceed prior"
    assert s_wrong.mastery >= 0, "Mastery must be non-negative"

def t_elo_update():
    from tutor.adaptive import EloSkillState
    e = EloSkillState()
    r0 = e.rating
    e.update(5, True)
    assert e.rating > r0
    e2 = EloSkillState()
    e2.update(5, False)
    assert e2.rating < r0

def t_learner_state():
    from tutor.adaptive import LearnerState
    from tutor import curriculum_loader as cl
    items = cl.load("data/T3.1_Math_Tutor/curriculum_full.json")
    state = LearnerState("test_smoke")
    for item in items[:15]:
        state.record_response(item, random.random() > 0.5)
    nxt = state.select_next_item(items, use_bkt=True)
    assert nxt is not None
    assert nxt["skill"] in ["counting","number_sense","addition","subtraction","word_problem"]

def t_serialise():
    from tutor.adaptive import LearnerState
    from tutor import curriculum_loader as cl
    items = cl.load("data/T3.1_Math_Tutor/curriculum_full.json")
    state = LearnerState("serialise_test")
    for item in items[:5]:
        state.record_response(item, True)
    d = state.to_dict()
    state2 = LearnerState.from_dict(d)
    assert state2.learner_id == "serialise_test"
    assert abs(state2.bkt["counting"].mastery - state.bkt["counting"].mastery) < 1e-6

test("BKT posterior update (correct)", t_bkt_update)
test("Elo rating update", t_elo_update)
test("LearnerState select_next_item", t_learner_state)
test("LearnerState serialise / deserialise", t_serialise)


# ------------------------------------------------------------------
print("\n=== lang_detect ===")

def t_pure_langs():
    from tutor.lang_detect import detect
    cases = [("five", "en"), ("neuf", "fr"), ("icyenda", "kin")]
    for text, expected in cases:
        dom, _ = detect(text)
        assert dom == expected, f"'{text}' → {dom}, expected {expected}"

def t_mix():
    from tutor.lang_detect import detect
    dom, scores = detect("neuf gatatu")
    assert dom == "mix", f"Expected mix, got {dom}"
    assert scores["fr"] > 0.1 and scores["kin"] > 0.1

def t_reply_lang():
    from tutor.lang_detect import reply_lang
    assert reply_lang("en") == "en"
    assert reply_lang("mix", fallback="fr") == "fr"

test("pure language detection EN/FR/KIN", t_pure_langs)
test("mixed language detection", t_mix)
test("reply_lang for mix", t_reply_lang)


# ------------------------------------------------------------------
print("\n=== asr_adapt (no Whisper) ===")

def t_extract_integer():
    from tutor.asr_adapt import extract_integer
    assert extract_integer("five") == 5
    assert extract_integer("neuf") == 9
    assert extract_integer("icyenda") == 9
    assert extract_integer("3") == 3
    assert extract_integer("the answer is twelve") == 12
    assert extract_integer("rimwe") == 1

def t_silence():
    import numpy as np
    from tutor.asr_adapt import is_silence
    silent = np.zeros(16000, dtype=np.float32)
    loud = np.random.normal(0, 0.3, 16000).astype(np.float32)
    assert is_silence(silent)
    assert not is_silence(loud)

test("extract_integer EN/FR/KIN number words", t_extract_integer)
test("silence detection", t_silence)


# ------------------------------------------------------------------
print("\n=== visual_grounding ===")

def t_blob_count():
    from tutor.visual_grounding import render_counting_stimulus, count_objects
    for n in range(1, 11):
        img = render_counting_stimulus(n)
        count, backend = count_objects(img, backend="blob")
        assert count == n, f"n={n}: blob counter returned {count}"

def t_render_shape():
    from tutor.visual_grounding import render_counting_stimulus
    import numpy as np
    img = render_counting_stimulus(5)
    assert img.shape == (128, 128, 3)
    assert img.dtype == np.uint8

def t_render_zero():
    from tutor.visual_grounding import render_counting_stimulus
    img = render_counting_stimulus(0)  # must not ZeroDivisionError
    assert img is not None

test("blob counter accuracy n=1–10", t_blob_count)
test("rendered stimulus shape & dtype", t_render_shape)
test("render_counting_stimulus(0) does not crash", t_render_zero)


# ------------------------------------------------------------------
print("\n=== model_loader (template mode) ===")

def t_templates():
    from tutor.model_loader import generate_feedback
    for lang in ("en", "fr", "kin"):
        for correct in (True, False):
            fb = generate_feedback(correct, 5, lang, "five")
            assert isinstance(fb, str) and len(fb) > 5, f"Empty feedback for lang={lang}"

test("template feedback all 3 languages × correct/wrong", t_templates)


# ------------------------------------------------------------------
print("\n=== progress_store ===")

def t_store():
    from tutor.progress_store import ProgressStore
    from tutor.adaptive import LearnerState
    from tutor import curriculum_loader as cl
    items = cl.load("data/T3.1_Math_Tutor/curriculum_full.json")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        store = ProgressStore(db_path, password="smoke-test")
        store.add_learner("smoke_user", "Smoke User", pin="0000")
        assert store.verify_pin("smoke_user", "0000")
        assert not store.verify_pin("smoke_user", "9999")

        state = LearnerState("smoke_user")
        sess_id = store.start_session("smoke_user", state.to_dict(), "en")
        for item in items[:6]:
            correct = random.random() > 0.5
            state.record_response(item, correct)
            store.log_response("smoke_user", sess_id, item["id"], item["skill"],
                               item.get("difficulty", 5), correct, latency_ms=900)
        store.end_session(sess_id, state.to_dict())

        report = store.weekly_report("smoke_user")
        assert report["sessions"] >= 1
        assert set(report["skills"].keys()) == {"counting","number_sense","addition","subtraction","word_problem"}

        dp = store.dp_sync_payload(epsilon=1.0)
        assert "noisy_skill_accuracy" in dp
        assert dp["epsilon_used"] == 1.0

        store.close()
    finally:
        os.unlink(db_path)

test("progress store (encrypted SQLite + DP sync)", t_store)


# ------------------------------------------------------------------
print("\n=== parent_report ===")

def t_html_report():
    from parent_report import generate_html
    dummy_report = {
        "learner_id": "amani",
        "week_starting": "2026-04-21",
        "sessions": 3,
        "skills": {s: {"current": 0.6, "delta": 0.1, "weekly_attempts": 5, "weekly_accuracy": 0.6}
                   for s in ["counting","number_sense","addition","subtraction","word_problem"]},
        "icons_for_parent": {"overall_arrow": "up", "best_skill": "counting", "needs_help": "subtraction"},
        "voiced_summary_audio": "tts/reports/amani_week.wav",
    }
    for lang in ("en", "fr", "kin"):
        html = generate_html(dummy_report, lang=lang)
        assert "<html" in html and "amani" in html

test("HTML parent report all 3 languages", t_html_report)


# ------------------------------------------------------------------
print()
if errors:
    print(f"\033[91m{len(errors)} test(s) FAILED:\033[0m")
    for name, tb in errors:
        print(f"\n--- {name} ---\n{tb}")
    sys.exit(1)
else:
    total = 19
    print(f"\033[92mAll tests passed ({total}/{total}) ✓\033[0m")
