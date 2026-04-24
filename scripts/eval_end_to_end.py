"""
eval_end_to_end.py — Single-command evaluation orchestrator.

Runs every eval script, measures p50 / p95 latency for each pipeline stage,
and aggregates all results into:

    figures/metrics.json   (machine-readable, everything)
    figures/metrics.md     (human-readable, the table dropped into the README)

Run:
    python3 scripts/eval_end_to_end.py
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------
# Stage latency benchmarks
# ------------------------------------------------------------------
def bench_template_feedback(n: int = 200) -> dict:
    from tutor.model_loader import generate_feedback
    timings = []
    for i in range(n):
        lang = ["en", "fr", "kin"][i % 3]
        correct = (i % 2) == 0
        t0 = time.perf_counter()
        generate_feedback(correct, (i % 20), lang, "five")
        timings.append((time.perf_counter() - t0) * 1000)
    return _stats(timings, n)


def bench_visual_grounding(n: int = 50) -> dict:
    from tutor.visual_grounding import render_counting_stimulus, count_objects
    timings_render, timings_count = [], []
    for i in range(n):
        k = (i % 10) + 1
        t0 = time.perf_counter()
        img = render_counting_stimulus(k)
        timings_render.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        count_objects(img, backend="blob")
        timings_count.append((time.perf_counter() - t0) * 1000)
    return {
        "render": _stats(timings_render, n),
        "blob_count": _stats(timings_count, n),
    }


def bench_bkt_update(n: int = 2000) -> dict:
    from tutor.adaptive import BKTSkillState
    s = BKTSkillState()
    timings = []
    for i in range(n):
        t0 = time.perf_counter()
        s.update(bool(i % 2))
        timings.append((time.perf_counter() - t0) * 1000)
    return _stats(timings, n)


def bench_adaptive_select(n: int = 200) -> dict:
    from tutor.adaptive import LearnerState
    from tutor import curriculum_loader as cl
    items = cl.load("data/T3.1_Math_Tutor/curriculum_full.json")
    state = LearnerState("bench")
    # warm up with 10 responses so mastery varies
    for it in items[:10]:
        state.record_response(it, True)
    timings = []
    for _ in range(n):
        t0 = time.perf_counter()
        state.select_next_item(items, use_bkt=True)
        timings.append((time.perf_counter() - t0) * 1000)
    return _stats(timings, n)


def bench_lang_detect(n: int = 2000) -> dict:
    from tutor.lang_detect import detect
    samples = [
        "five", "the answer is seven", "cinq", "la réponse est neuf",
        "rimwe", "ni kabiri", "neuf gatatu", "yes yego", "three trois",
        "je dis icumi",
    ]
    timings = []
    for i in range(n):
        t0 = time.perf_counter()
        detect(samples[i % len(samples)])
        timings.append((time.perf_counter() - t0) * 1000)
    return _stats(timings, n)


def bench_e2e_scoring(n: int = 200) -> dict:
    """End-to-end: lang_detect + extract_integer + BKT update + template feedback."""
    from tutor.adaptive import LearnerState
    from tutor.asr_adapt import extract_integer
    from tutor.lang_detect import detect
    from tutor.model_loader import generate_feedback
    from tutor import curriculum_loader as cl
    items = cl.load("data/T3.1_Math_Tutor/curriculum_full.json")
    texts = ["five", "neuf", "gatatu", "ten", "dix", "icumi", "twelve", "cinq", "seven", "rimwe"]
    state = LearnerState("bench_e2e")
    timings = []
    for i in range(n):
        item = items[i % len(items)]
        text = texts[i % len(texts)]
        t0 = time.perf_counter()
        lang, _ = detect(text)
        n_int = extract_integer(text)
        correct = (n_int == item["answer_int"])
        state.record_response(item, correct)
        generate_feedback(correct, item["answer_int"], lang if lang in ("en", "fr", "kin") else "en", text)
        timings.append((time.perf_counter() - t0) * 1000)
    return _stats(timings, n)


def _stats(timings_ms, n) -> dict:
    a = np.array(timings_ms)
    return {
        "n": n,
        "mean_ms": round(float(a.mean()), 4),
        "p50_ms": round(float(np.percentile(a, 50)), 4),
        "p95_ms": round(float(np.percentile(a, 95)), 4),
        "p99_ms": round(float(np.percentile(a, 99)), 4),
        "max_ms": round(float(a.max()), 4),
    }


# ------------------------------------------------------------------
# Script runner
# ------------------------------------------------------------------
def run_script(script: str) -> dict:
    path = Path("scripts") / script
    print(f"\n>>> running {path} ...")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True, text=True, cwd=str(Path.cwd()),
    )
    elapsed = time.time() - t0
    print(proc.stdout[-500:] if proc.stdout else "")
    if proc.returncode != 0:
        print(f"  stderr: {proc.stderr[-500:]}")
    return {
        "script": script,
        "returncode": proc.returncode,
        "elapsed_s": round(elapsed, 2),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # 1. Run sub-evals
    ran = []
    for scr in ("eval_bkt.py", "eval_asr.py", "eval_feedback.py"):
        ran.append(run_script(scr))

    # 2. Load their outputs
    loaded = {}
    for name, fn in [
        ("bkt", "bkt_metrics.json"),
        ("asr", "asr_metrics.json"),
        ("feedback", "feedback_metrics.json"),
    ]:
        p = FIG_DIR / fn
        if p.exists():
            loaded[name] = json.loads(p.read_text())

    # 3. Latency benchmarks on this machine
    print("\n>>> latency benchmarks ...")
    latency = {
        "lang_detect": bench_lang_detect(),
        "bkt_update": bench_bkt_update(),
        "adaptive_select": bench_adaptive_select(),
        "visual_grounding": bench_visual_grounding(),
        "template_feedback": bench_template_feedback(),
        "end_to_end_scoring": bench_e2e_scoring(),
    }
    for stage, s in latency.items():
        head = s if "p95_ms" in s else s.get("blob_count", s)
        print(f"  {stage:24s}  p50={head.get('p50_ms','–')}ms  "
              f"p95={head.get('p95_ms','–')}ms  n={head.get('n','–')}")

    # 4. Aggregate
    agg = {
        "environment": env,
        "sub_eval_runs": ran,
        "latency_ms": latency,
        "bkt": loaded.get("bkt"),
        "asr": loaded.get("asr"),
        "feedback": loaded.get("feedback"),
    }
    out_json = FIG_DIR / "metrics.json"
    out_json.write_text(json.dumps(agg, indent=2, ensure_ascii=False))
    print(f"\n[json] wrote {out_json}")

    # 5. Markdown
    write_markdown(agg)


def write_markdown(agg: dict):
    bkt = agg.get("bkt") or {}
    asr = agg.get("asr") or {}
    fb = agg.get("feedback") or {}

    lines = []
    lines.append("# AI Math Tutor — Reported Metrics\n")
    lines.append(f"_Machine_: `{agg['environment']['platform']}`  ")
    lines.append(f"_Python_: `{agg['environment']['python']}`  ")
    lines.append(f"_Generated_: `{agg['environment']['timestamp']}`\n")

    # Headline
    lines.append("## Headline\n")
    lines.append("| Layer | Metric | Value | 95% CI / notes |")
    lines.append("|---|---|---|---|")
    if bkt:
        m = bkt["models"]["BKT"]
        e = bkt["models"]["Elo"]
        pr = bkt["models"]["Prior-only"]
        lines.append(f"| **Knowledge Tracing** | BKT AUC (held-out) | **{m['auc']:.4f}** | "
                     f"95% CI [{m['auc_ci95'][0]:.4f}, {m['auc_ci95'][1]:.4f}] |")
        lines.append(f"| Knowledge Tracing | Elo baseline AUC | {e['auc']:.4f} | Δ = "
                     f"{m['auc']-e['auc']:+.4f} |")
        lines.append(f"| Knowledge Tracing | Prior-only baseline AUC | {pr['auc']:.4f} | "
                     f"(no learning) |")
        lines.append(f"| Knowledge Tracing | Brier score (BKT) | {m['brier']:.4f} | lower is better |")
        lines.append(f"| Knowledge Tracing | Log-loss (BKT) | {m['log_loss']:.4f} | lower is better |")
    if asr:
        p = asr["number_word_parser"]
        l = asr["language_detection"]
        ps = asr["pitch_shift"]
        lines.append(f"| **ASR number parser** | Accuracy (EN/FR/KIN, n=180) | **{p['accuracy']*100:.1f}%** | 60 per language |")
        lines.append(f"| **Language detection** | Accuracy (n=120) | **{l['accuracy']*100:.1f}%** | 30 each EN/FR/KIN/mix |")
        for lang in ("en", "fr", "kin", "mix"):
            v = l["per_language"][lang]
            lines.append(f"| Language detection | {lang} F1 | {v['f1']:.3f} | P={v['precision']:.3f} R={v['recall']:.3f} |")
        lines.append(f"| Pitch-shift normaliser | Energy ratio | {ps['mean_energy_ratio']:.3f} | target ≈ 1.0, stable |")
    if fb:
        t = fb["template"]
        lines.append(f"| **Feedback (template)** | Rubric (0–6) | **{t['rubric_mean_0_6']}** | "
                     f"≥ 5/6 on {t['rubric_pass_rate_5of6']*100:.1f}% of 90 cases |")
        lines.append(f"| Feedback (template) | BLEU-2 | {t['bleu2_mean']:.3f} | vs. 3 gold refs per cell |")
        lines.append(f"| Feedback (template) | ROUGE-L | {t['rouge_l_mean']:.3f} | — |")
        lines.append(f"| Feedback (template) | p95 latency | {t['latency_ms']['p95']} ms | p50 {t['latency_ms']['p50']} ms |")
    lines.append("")

    # Latency
    lines.append("## Latency (per-stage, measured on run machine)\n")
    lines.append("| Stage | p50 ms | p95 ms | p99 ms | N |")
    lines.append("|---|---|---|---|---|")
    for stage, s in agg["latency_ms"].items():
        if stage == "visual_grounding":
            c = s["blob_count"]
            r = s["render"]
            lines.append(f"| visual_grounding (render) | {r['p50_ms']} | {r['p95_ms']} | {r['p99_ms']} | {r['n']} |")
            lines.append(f"| visual_grounding (blob count) | {c['p50_ms']} | {c['p95_ms']} | {c['p99_ms']} | {c['n']} |")
        else:
            lines.append(f"| {stage} | {s['p50_ms']} | {s['p95_ms']} | {s['p99_ms']} | {s['n']} |")
    lines.append("")

    # BKT per-skill
    if bkt:
        lines.append("## BKT per-skill breakdown\n")
        lines.append("| Skill | N | AUC | Precision | Recall | F1 |")
        lines.append("|---|---|---|---|---|---|")
        for sk, v in bkt["models"]["BKT"]["per_skill"].items():
            lines.append(f"| {sk} | {v['n']} | {v['auc']} | {v['precision']} | {v['recall']} | {v['f1']} |")
        lines.append("")

    # Reproduce
    lines.append("## Reproducibility\n")
    lines.append("Every number in this document can be regenerated with:\n")
    lines.append("```bash\npython3 scripts/eval_end_to_end.py\n```")
    lines.append("which runs `eval_bkt.py`, `eval_asr.py`, `eval_feedback.py` and the "
                 "latency micro-benchmarks, writing `figures/metrics.json` and this file "
                 "(`figures/metrics.md`).")
    lines.append("")
    lines.append("### Optional WER on real audio\n")
    lines.append("```bash\npython3 scripts/eval_asr.py --audio-csv your_cv_test.csv\n```")
    lines.append("CSV columns: `audio_path,reference,language` (e.g. Common Voice KIN).")

    out = FIG_DIR / "metrics.md"
    out.write_text("\n".join(lines))
    print(f"[md] wrote {out}")


if __name__ == "__main__":
    main()
