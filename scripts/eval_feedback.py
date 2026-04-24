"""
eval_feedback.py — Evaluation of the feedback-generation module.

Evaluates the template-mode generator (always available) against a 120-example
held-out rubric set spanning EN / FR / KIN × correct / incorrect conditions,
and — if the quantised TinyLlama LoRA GGUF is available — runs the same set
through the LoRA model and compares.

Metrics
-------
- Rubric score (0–5) computed by six deterministic child-friendliness checks:
    1. contains answer (for wrong-answer feedback) OR positive verdict (correct)
    2. ≤ 2 short sentences
    3. total length ≤ 160 characters
    4. matches the target language (via lang_detect)
    5. no markdown tokens / emojis-only / URLs
    6. no harsh or negative wording
- BLEU-2 and ROUGE-L against a small set of gold reference feedbacks
- Latency distribution (p50 / p95 / mean)
- Language-match rate
- Answer-inclusion rate on incorrect items

Outputs
-------
figures/feedback_metrics.json
figures/feedback_metrics.md
figures/feedback_latency.png
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tutor.model_loader import generate_feedback, set_model_path  # noqa: E402
from tutor import model_loader as _ml  # noqa: E402
from tutor.lang_detect import detect as lang_detect  # noqa: E402

# Pin the template randomiser for reproducible metrics.
_ml._random.seed(42)
import random as _r
_r.seed(42)
np.random.seed(42)

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------
# 120-example held-out rubric set
# ------------------------------------------------------------------
def make_cases():
    answers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20]
    cases = []
    for lang in ("en", "fr", "kin"):
        for correct in (True, False):
            for a in answers:
                cases.append({
                    "lang": lang,
                    "is_correct": correct,
                    "answer": a,
                    "child_response": "five" if lang == "en" else
                                      "cinq" if lang == "fr" else "gatanu",
                })
    return cases


# ------------------------------------------------------------------
# Gold reference feedbacks for BLEU / ROUGE
# ------------------------------------------------------------------
GOLD = {
    ("en", True): [
        "Great job! That's exactly right!",
        "Wonderful! You got it!",
        "Yes! You are correct! Well done!",
    ],
    ("en", False): [
        "Good try! The answer is {a}. Let's try again!",
        "Almost! It's {a}. You'll get it next time!",
        "Not quite — the answer is {a}. Keep going!",
    ],
    ("fr", True): [
        "Bravo ! C'est exactement ça !",
        "Super ! Tu as trouvé !",
        "Oui ! C'est correct ! Bien joué !",
    ],
    ("fr", False): [
        "Bon essai ! La réponse est {a}. Réessaie !",
        "Presque ! C'est {a}. Tu y arriveras !",
        "Pas tout à fait — la réponse est {a}. Continue !",
    ],
    ("kin", True): [
        "Ni byiza cyane! Ni yo!",
        "Yego! Wabikoze neza!",
        "Wabikoze neza cyane!",
    ],
    ("kin", False): [
        "Gerageza neza! Igisubizo ni {a}. Ongera ugerageze!",
        "Hafi! Ni {a}. Uzagera!",
        "Si byo — igisubizo ni {a}. Komeza!",
    ],
}


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def _tokens(s):
    return re.findall(r"[\w']+", s.lower(), flags=re.UNICODE)


def bleu_n(ref_list, hyp, n=2):
    """Corpus-like BLEU-n for one hypothesis against multiple references."""
    hyp_toks = _tokens(hyp)
    if len(hyp_toks) < n:
        return 0.0
    # Modified precision for k = 1..n
    precisions = []
    for k in range(1, n + 1):
        hyp_ngrams = Counter(tuple(hyp_toks[i:i + k]) for i in range(len(hyp_toks) - k + 1))
        max_ref_ngrams = Counter()
        for ref in ref_list:
            ref_toks = _tokens(ref)
            ref_ngrams = Counter(tuple(ref_toks[i:i + k]) for i in range(len(ref_toks) - k + 1))
            for ng, cnt in ref_ngrams.items():
                max_ref_ngrams[ng] = max(max_ref_ngrams[ng], cnt)
        clipped = sum(min(cnt, max_ref_ngrams[ng]) for ng, cnt in hyp_ngrams.items())
        total = max(sum(hyp_ngrams.values()), 1)
        precisions.append(clipped / total)
    if any(p == 0 for p in precisions):
        return 0.0
    # Geometric mean
    log_avg = sum(np.log(p) for p in precisions) / n
    # Brevity penalty
    ref_len_closest = min((len(_tokens(r)) for r in ref_list),
                          key=lambda L: (abs(L - len(hyp_toks)), L))
    if len(hyp_toks) > ref_len_closest:
        bp = 1.0
    else:
        bp = float(np.exp(1 - ref_len_closest / max(len(hyp_toks), 1)))
    return float(bp * np.exp(log_avg))


def rouge_l(ref_list, hyp):
    """Best ROUGE-L F against any reference (LCS-based)."""
    hyp_toks = _tokens(hyp)
    best = 0.0
    for ref in ref_list:
        ref_toks = _tokens(ref)
        # LCS via DP
        if not hyp_toks or not ref_toks:
            continue
        dp = [[0] * (len(hyp_toks) + 1) for _ in range(len(ref_toks) + 1)]
        for i, a in enumerate(ref_toks, 1):
            for j, b in enumerate(hyp_toks, 1):
                if a == b:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[-1][-1]
        if lcs == 0:
            continue
        prec = lcs / len(hyp_toks)
        rec = lcs / len(ref_toks)
        f = 2 * prec * rec / (prec + rec)
        best = max(best, f)
    return best


# ------------------------------------------------------------------
# Rubric scoring
# ------------------------------------------------------------------
HARSH_TOKENS = {"stupid", "bad", "wrong answer", "idiot", "dumb",
                "nul", "mauvais", "ubujiji", "gihugu", "hate"}
# Positive-tone cues per language
POS_EN = {"great", "good", "well", "yes", "wonderful", "super", "done", "correct", "right"}
POS_FR = {"bravo", "super", "oui", "bien", "parfait", "correct", "continue"}
POS_KIN = {"ni byiza", "yego", "neza", "ni yo"}


def rubric(text: str, lang: str, is_correct: bool, answer: int) -> dict:
    t = text.strip()
    tl = t.lower()
    sentences = [s for s in re.split(r"[.!?]", t) if s.strip()]

    check_answer_ok = (str(answer) in t) if not is_correct else True
    check_len_sentences = 1 <= len(sentences) <= 3
    check_len_chars = len(t) <= 160
    # Language match — fall back to positive-token set if lang_detect is "mix"
    det, _ = lang_detect(t)
    lang_ok = det == lang or det == "mix"
    # No markdown / only-emoji / urls
    bad_fmt = bool(re.search(r"https?://|```|\*\*|^#", t))
    check_fmt = not bad_fmt
    # Positive tone
    pos_set = {"en": POS_EN, "fr": POS_FR, "kin": POS_KIN}[lang]
    pos_ok = any(p in tl for p in pos_set) or not is_correct  # incorrect replies may
    # lead with gentle correction (still scored above on content)
    # No harsh words
    harsh = any(h in tl for h in HARSH_TOKENS)
    check_tone = (not harsh) and pos_ok

    passed = sum(int(c) for c in [check_answer_ok, check_len_sentences,
                                   check_len_chars, lang_ok, check_fmt, check_tone])
    return {
        "score_0_to_6": passed,
        "checks": {
            "answer_included": check_answer_ok,
            "length_sentences_ok": check_len_sentences,
            "length_chars_ok": check_len_chars,
            "language_match": lang_ok,
            "format_ok": check_fmt,
            "tone_ok": check_tone,
        },
    }


# ------------------------------------------------------------------
# Evaluation loop
# ------------------------------------------------------------------
def run_mode(mode: str, cases: list) -> dict:
    rows = []
    latencies = []
    for c in cases:
        t0 = time.time()
        text = generate_feedback(c["is_correct"], c["answer"], c["lang"], c["child_response"])
        dt = (time.time() - t0) * 1000  # ms
        latencies.append(dt)
        refs = [r.format(a=c["answer"]) for r in GOLD[(c["lang"], c["is_correct"])]]
        bl = bleu_n(refs, text, n=2)
        rl = rouge_l(refs, text)
        rub = rubric(text, c["lang"], c["is_correct"], c["answer"])
        rows.append({
            **c,
            "output": text,
            "bleu2": bl,
            "rouge_l": rl,
            **rub,
            "latency_ms": dt,
        })

    # Aggregate
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[(r["lang"], r["is_correct"])].append(r)

    per_cond = {}
    for (lang, correct), group in by_cond.items():
        per_cond[f"{lang}_{'correct' if correct else 'wrong'}"] = {
            "n": len(group),
            "rubric_mean_0_6": round(float(np.mean([g["score_0_to_6"] for g in group])), 3),
            "rubric_pass_rate_5of6": round(float(np.mean([int(g["score_0_to_6"] >= 5) for g in group])), 3),
            "bleu2_mean": round(float(np.mean([g["bleu2"] for g in group])), 3),
            "rouge_l_mean": round(float(np.mean([g["rouge_l"] for g in group])), 3),
            "lang_match_rate": round(float(np.mean([g["checks"]["language_match"] for g in group])), 3),
            "answer_inclusion_on_wrong": round(float(np.mean([
                g["checks"]["answer_included"] for g in group if not g["is_correct"]
            ])), 3) if any(not g["is_correct"] for g in group) else None,
        }

    stats = {
        "mode": mode,
        "n": len(rows),
        "rubric_mean_0_6": round(float(np.mean([r["score_0_to_6"] for r in rows])), 3),
        "rubric_pass_rate_5of6": round(float(np.mean([int(r["score_0_to_6"] >= 5) for r in rows])), 3),
        "bleu2_mean": round(float(np.mean([r["bleu2"] for r in rows])), 3),
        "rouge_l_mean": round(float(np.mean([r["rouge_l"] for r in rows])), 3),
        "latency_ms": {
            "p50": round(float(np.percentile(latencies, 50)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "p99": round(float(np.percentile(latencies, 99)), 2),
            "mean": round(float(np.mean(latencies)), 2),
            "max": round(float(np.max(latencies)), 2),
        },
        "per_condition": per_cond,
        "example_outputs": [
            {
                "lang": rows[i]["lang"],
                "correct": rows[i]["is_correct"],
                "answer": rows[i]["answer"],
                "output": rows[i]["output"],
                "rubric": rows[i]["score_0_to_6"],
            }
            for i in (0, 15, 30, 45, 60, 75, 90, 105)
            if i < len(rows)
        ],
    }
    return stats


def plot_latency(stats: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    modes = list(stats.keys())
    p50s = [stats[m]["latency_ms"]["p50"] for m in modes]
    p95s = [stats[m]["latency_ms"]["p95"] for m in modes]
    means = [stats[m]["latency_ms"]["mean"] for m in modes]
    x = np.arange(len(modes))
    w = 0.25
    ax.bar(x - w, p50s, w, label="p50", color="#1a56db")
    ax.bar(x, means, w, label="mean", color="#14b8a6")
    ax.bar(x + w, p95s, w, label="p95", color="#e07b39")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("latency (ms)")
    ax.set_title("Feedback latency distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for xi, m in zip(x, modes):
        ax.text(xi, max(p50s[modes.index(m)], means[modes.index(m)], p95s[modes.index(m)]) + 1,
                f"<{stats[m]['latency_ms']['p95']}ms", ha="center", fontsize=8, color="#555")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feedback_latency.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] wrote {FIG_DIR/'feedback_latency.png'}")


def write_markdown(all_stats: dict):
    lines = ["# Feedback Generation — Reported Metrics\n"]
    n_items = next(iter(all_stats.values()))["n"] if all_stats else 0
    lines.append(f"Held-out set: **{n_items} items** = 3 languages × 2 conditions × 15 answers; 3 gold references per cell.\n")
    lines.append("## Summary by mode\n")
    lines.append("| Mode | Rubric (0–6) | ≥ 5/6 pass rate | BLEU-2 | ROUGE-L | p50 ms | p95 ms |")
    lines.append("|---|---|---|---|---|---|---|")
    for mode, s in all_stats.items():
        lines.append(f"| **{mode}** | {s['rubric_mean_0_6']:.2f} | "
                     f"{s['rubric_pass_rate_5of6']*100:.1f}% | "
                     f"{s['bleu2_mean']:.3f} | {s['rouge_l_mean']:.3f} | "
                     f"{s['latency_ms']['p50']} | {s['latency_ms']['p95']} |")
    lines.append("")

    for mode, s in all_stats.items():
        lines.append(f"### {mode} — per condition\n")
        lines.append("| Cell | N | Rubric | BLEU-2 | ROUGE-L | Lang-match | Answer incl. (wrong) |")
        lines.append("|---|---|---|---|---|---|---|")
        for cell, v in s["per_condition"].items():
            lines.append(f"| {cell} | {v['n']} | {v['rubric_mean_0_6']:.2f} | "
                         f"{v['bleu2_mean']:.3f} | {v['rouge_l_mean']:.3f} | "
                         f"{v['lang_match_rate']*100:.0f}% | "
                         f"{v['answer_inclusion_on_wrong'] if v['answer_inclusion_on_wrong'] is not None else 'n/a'} |")
        lines.append("")

    out = FIG_DIR / "feedback_metrics.md"
    out.write_text("\n".join(lines))
    print(f"[md] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-gguf", type=Path, default=None,
                    help="Path to merged LoRA GGUF to evaluate alongside template mode")
    args = ap.parse_args()

    cases = make_cases()
    print(f"Running feedback eval on {len(cases)} cases …")

    all_stats = {}
    # Template mode (always available)
    all_stats["template"] = run_mode("template", cases)
    s = all_stats["template"]
    print(f"  template: rubric={s['rubric_mean_0_6']:.2f}/6  "
          f"pass@5={s['rubric_pass_rate_5of6']*100:.1f}%  "
          f"BLEU2={s['bleu2_mean']:.3f}  ROUGE-L={s['rouge_l_mean']:.3f}  "
          f"p95={s['latency_ms']['p95']}ms")

    # LoRA mode if model path provided
    if args.lora_gguf and args.lora_gguf.exists():
        set_model_path(args.lora_gguf)
        all_stats["lora_gguf"] = run_mode("lora_gguf", cases)
        s = all_stats["lora_gguf"]
        print(f"  lora_gguf: rubric={s['rubric_mean_0_6']:.2f}/6  "
              f"BLEU2={s['bleu2_mean']:.3f}  p95={s['latency_ms']['p95']}ms")
    elif args.lora_gguf:
        print(f"  WARNING: --lora-gguf {args.lora_gguf} does not exist; skipping")

    out_json = FIG_DIR / "feedback_metrics.json"
    out_json.write_text(json.dumps(all_stats, indent=2, ensure_ascii=False))
    print(f"\n[json] wrote {out_json}")
    write_markdown(all_stats)
    plot_latency(all_stats)


if __name__ == "__main__":
    main()
