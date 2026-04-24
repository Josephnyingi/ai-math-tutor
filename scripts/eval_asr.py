"""
eval_asr.py — Evaluation of the child-speech ASR pipeline.

Covers three layers of the ASR stack:

  1. Number-word parsing (extract_integer)
       - Deterministic, runs with zero model loads.
       - Reports per-language accuracy + confusion categories.

  2. Language detection (lang_detect.detect)
       - Evaluated on a labelled fixture of 120 EN/FR/KIN/mix utterances.
       - Reports overall accuracy + per-language precision / recall / F1.

  3. Whisper-tiny WER on real audio  (--audio-csv <path>)
       - Optional.  Requires openai-whisper + a CSV of
         (audio_path, reference, language).
       - Reports WER + CER per language and a child-pitch-correction
         ablation (± --4.5 semitones normalisation).

Outputs
-------
figures/asr_metrics.json   machine-readable metrics
figures/asr_metrics.md     markdown summary
figures/asr_confusion.png  number-word confusion matrix (if matplotlib)

Usage
-----
    # Offline, no model downloads — always works:
    python3 scripts/eval_asr.py

    # With a Common Voice KIN / FR / EN test CSV (real audio):
    python3 scripts/eval_asr.py --audio-csv data/asr_test.csv

CSV format for --audio-csv:
    audio_path,reference,language
    /path/to/foo.wav,"rimwe kabiri gatatu",kin
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tutor.asr_adapt import extract_integer, _pitch_normalise, is_silence  # noqa: E402
from tutor.lang_detect import detect as lang_detect  # noqa: E402

import numpy as np  # noqa: E402

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


# ======================================================================
# 1. Number-word parser eval
# ======================================================================
# 180 labelled utterances: 60 EN, 60 FR, 60 KIN — include common child variants,
# noisy prefixes, and filler words.
NUMBER_WORD_CASES = []


def _add(cases, utterances, gold, lang):
    for u in utterances:
        cases.append({"text": u, "gold": gold, "lang": lang})


# English
_add(NUMBER_WORD_CASES, ["five", "The answer is five", "um five", "5", "it's 5",
                         "five please", "i think five", "is it five?", "five!", "five five"], 5, "en")
_add(NUMBER_WORD_CASES, ["three", "three", "The answer is three", "3", "maybe 3",
                         "three right?", "its three", "uhh three", "answer three", "three three"], 3, "en")
_add(NUMBER_WORD_CASES, ["twelve", "twelve please", "12", "it is 12", "twelve",
                         "i say twelve", "um twelve", "twelve?", "the answer twelve", "answer is 12"], 12, "en")
_add(NUMBER_WORD_CASES, ["zero", "zero", "0", "nothing zero", "zero is right",
                         "its zero", "zero maybe", "hmm zero", "0 right", "zero"], 0, "en")
_add(NUMBER_WORD_CASES, ["ten", "ten", "10", "i say ten", "ten please",
                         "is it 10", "the answer ten", "10", "ten", "ten"], 10, "en")
_add(NUMBER_WORD_CASES, ["seven", "seven", "7", "seven please", "i think seven",
                         "uhh seven", "answer is 7", "seven?", "seven seven", "7"], 7, "en")

# French
_add(NUMBER_WORD_CASES, ["cinq", "cinq", "5", "je dis cinq", "c'est cinq",
                         "cinq peut être", "euh cinq", "cinq?", "la réponse est 5", "cinq"], 5, "fr")
_add(NUMBER_WORD_CASES, ["trois", "trois", "3", "trois s'il vous plaît",
                         "je pense trois", "euh trois", "la réponse trois", "3 peut-être", "trois", "c'est trois"], 3, "fr")
_add(NUMBER_WORD_CASES, ["neuf", "neuf", "9", "neuf", "c'est neuf",
                         "je dis neuf", "la réponse est neuf", "9", "neuf", "peut-être neuf"], 9, "fr")
_add(NUMBER_WORD_CASES, ["dix", "dix", "10", "dix", "la réponse dix",
                         "c'est 10", "dix", "je dis dix", "dix oui", "10"], 10, "fr")
_add(NUMBER_WORD_CASES, ["quatre", "quatre", "4", "c'est quatre", "quatre",
                         "euh quatre", "quatre?", "la réponse quatre", "4", "je dis quatre"], 4, "fr")
_add(NUMBER_WORD_CASES, ["douze", "douze", "12", "c'est douze",
                         "douze s'il vous plaît", "12", "je dis douze", "douze?", "douze", "la réponse douze"], 12, "fr")

# Kinyarwanda
_add(NUMBER_WORD_CASES, ["rimwe", "ni rimwe", "1", "rimwe",
                         "rimwe koko", "nkibona rimwe", "rimwe?", "yego rimwe", "rimwe rimwe", "ni 1"], 1, "kin")
_add(NUMBER_WORD_CASES, ["kabiri", "ni kabiri", "2", "kabiri",
                         "kabiri koko", "nkibona kabiri", "kabiri?", "yego kabiri", "kabiri kabiri", "ni 2"], 2, "kin")
_add(NUMBER_WORD_CASES, ["gatatu", "ni gatatu", "3", "gatatu",
                         "gatatu koko", "yego gatatu", "gatatu?", "nkibona gatatu", "gatatu gatatu", "ni 3"], 3, "kin")
_add(NUMBER_WORD_CASES, ["gatanu", "ni gatanu", "5", "gatanu",
                         "gatanu koko", "yego gatanu", "gatanu?", "nkibona gatanu", "gatanu gatanu", "ni 5"], 5, "kin")
_add(NUMBER_WORD_CASES, ["icyenda", "ni icyenda", "9", "icyenda",
                         "icyenda koko", "yego icyenda", "icyenda?", "nkibona icyenda", "icyenda icyenda", "ni 9"], 9, "kin")
_add(NUMBER_WORD_CASES, ["icumi", "ni icumi", "10", "icumi",
                         "icumi koko", "yego icumi", "icumi?", "nkibona icumi", "icumi icumi", "ni 10"], 10, "kin")


def eval_number_word_parser() -> dict:
    by_lang = defaultdict(lambda: {"correct": 0, "total": 0, "misses": []})
    overall_correct = 0
    confusion = Counter()  # (gold, pred) counts
    for case in NUMBER_WORD_CASES:
        pred = extract_integer(case["text"])
        bl = by_lang[case["lang"]]
        bl["total"] += 1
        if pred == case["gold"]:
            bl["correct"] += 1
            overall_correct += 1
        else:
            bl["misses"].append({"text": case["text"], "gold": case["gold"], "pred": pred})
        confusion[(case["gold"], pred)] += 1

    return {
        "total": len(NUMBER_WORD_CASES),
        "correct": overall_correct,
        "accuracy": round(overall_correct / len(NUMBER_WORD_CASES), 4),
        "per_language": {
            lang: {
                "n": v["total"],
                "correct": v["correct"],
                "accuracy": round(v["correct"] / v["total"], 4),
                "example_misses": v["misses"][:3],
            }
            for lang, v in by_lang.items()
        },
        "confusion_non_match": {
            f"{g}->{p}": c for (g, p), c in confusion.items() if g != p
        },
    }


# ======================================================================
# 2. Language detection eval
# ======================================================================
LANG_CASES = []

# Pure English (30)
for u in ["five", "the answer is seven", "how many are there", "three apples",
         "ten children", "hello teacher", "i think nine", "yes it's eight",
         "the number is four", "two plus two", "count to ten", "one banana",
         "six dogs here", "eleven please", "i don't know", "maybe twelve",
         "the answer", "what is that", "fifteen cars", "thirteen birds",
         "zero things", "fourteen", "nine nine", "thirty", "twenty cats",
         "how many children", "five and three", "please help me", "the big number",
         "yes"]:
    LANG_CASES.append({"text": u, "gold": "en"})

# Pure French (30)
for u in ["cinq", "la réponse est sept", "combien y en a-t-il", "trois pommes",
         "dix enfants", "bonjour maîtresse", "je pense neuf", "oui c'est huit",
         "le nombre est quatre", "deux plus deux", "compter jusqu'à dix", "une banane",
         "six chiens ici", "onze s'il vous plaît", "je ne sais pas", "peut-être douze",
         "la réponse", "qu'est-ce que c'est", "quinze voitures", "treize oiseaux",
         "zéro choses", "quatorze", "neuf neuf", "trente", "vingt chats",
         "combien d'enfants", "cinq et trois", "aidez-moi s'il vous plaît",
         "le grand nombre", "oui"]:
    LANG_CASES.append({"text": u, "gold": "fr"})

# Pure Kinyarwanda (30)
for u in ["rimwe", "igisubizo ni indwi", "ni bangahe", "ibiyobyabwenge bitatu",
         "abana icumi", "muraho mwarimu", "ndatekereza icyenda", "yego ni umunani",
         "umubare ni kane", "kabiri na kabiri", "bara kugera kuri icumi", "ibitoki rimwe",
         "imbwa gatandatu", "cumi na rimwe", "simbizi", "wenda cumi na kabiri",
         "igisubizo", "ni iki", "imodoka cumi na gatanu", "inyoni cumi na gatatu",
         "zeru", "cumi na kane", "icyenda icyenda", "makumyabiri na gatatu",
         "injangwe makumyabiri", "abana bangahe", "gatanu na gatatu", "mfasha",
         "umubare munini", "yego"]:
    LANG_CASES.append({"text": u, "gold": "kin"})

# Mix (30) — intentionally code-switched
for u in ["neuf gatatu", "five cinq", "rimwe one", "two kabiri", "gatatu three",
         "yes yego", "oui rimwe", "sept icyenda", "huit icumi", "neuf kane",
         "dix ten", "the answer cinq", "is it rimwe", "trois three",
         "i think neuf", "je dis five", "cinq five", "kabiri deux", "yego oui",
         "umunani eight", "icumi dix", "five rimwe", "two trois", "one rimwe deux",
         "je pense kabiri", "answer gatatu", "rimwe un", "three gatatu",
         "deux kabiri", "nine icyenda"]:
    LANG_CASES.append({"text": u, "gold": "mix"})


def eval_language_detection() -> dict:
    by_gold = defaultdict(lambda: {"tp": 0, "fn": 0})
    by_pred = defaultdict(lambda: {"fp": 0})
    correct = 0
    langs = ("en", "fr", "kin", "mix")
    confusion = {g: Counter() for g in langs}
    examples_wrong = []

    for case in LANG_CASES:
        pred, _ = lang_detect(case["text"])
        g = case["gold"]
        confusion[g][pred] += 1
        if pred == g:
            correct += 1
            by_gold[g]["tp"] += 1
        else:
            by_gold[g]["fn"] += 1
            by_pred[pred]["fp"] += 1
            if len(examples_wrong) < 10:
                examples_wrong.append({"text": case["text"], "gold": g, "pred": pred})

    per_lang = {}
    for g in langs:
        tp = by_gold[g]["tp"]
        fn = by_gold[g]["fn"]
        fp = by_pred[g]["fp"]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        per_lang[g] = {
            "n": tp + fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    return {
        "total": len(LANG_CASES),
        "accuracy": round(correct / len(LANG_CASES), 4),
        "per_language": per_lang,
        "confusion_matrix": {g: dict(c) for g, c in confusion.items()},
        "example_errors": examples_wrong,
    }


# ======================================================================
# 3. Pitch-shift stability
# ======================================================================
def eval_pitch_shift() -> dict:
    """Verify the child-pitch normaliser is numerically stable and
    preserves a sinusoid's energy within tolerance."""
    sr = 16000
    durations = [0.2, 0.5, 1.0, 2.0]
    stats = []
    for d in durations:
        n = int(sr * d)
        t = np.arange(n) / sr
        # 400 Hz sine wave + small noise
        audio = 0.5 * np.sin(2 * np.pi * 400 * t).astype(np.float32)
        audio += 0.01 * np.random.default_rng(0).standard_normal(n).astype(np.float32)
        shifted = _pitch_normalise(audio, semitones=-4.5)
        assert shifted.shape == audio.shape
        energy_in = float(np.mean(audio ** 2))
        energy_out = float(np.mean(shifted ** 2))
        ratio = energy_out / max(energy_in, 1e-12)
        stats.append({
            "duration_s": d,
            "n_samples": n,
            "energy_ratio": round(ratio, 4),
            "max_abs_out": round(float(np.max(np.abs(shifted))), 4),
            "nan_or_inf": bool(np.any(~np.isfinite(shifted))),
        })
    mean_ratio = float(np.mean([s["energy_ratio"] for s in stats]))
    return {
        "cases": stats,
        "mean_energy_ratio": round(mean_ratio, 4),
        "semitones": -4.5,
        "all_stable": all(not s["nan_or_inf"] for s in stats),
    }


# ======================================================================
# 4. WER helpers (Levenshtein over tokens)
# ======================================================================
def _tokenise(s: str) -> list:
    s = unicodedata.normalize("NFC", s.lower())
    return re.findall(r"[a-záàâéèêëîïôùûüçñ0-9]+", s)


def edit_distance(a: list, b: list) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref = _tokenise(reference)
    hyp = _tokenise(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def cer(reference: str, hypothesis: str) -> float:
    ref = list(unicodedata.normalize("NFC", reference.lower().strip()))
    hyp = list(unicodedata.normalize("NFC", hypothesis.lower().strip()))
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


# ======================================================================
# 5. Optional Whisper WER path
# ======================================================================
def eval_whisper_wer(audio_csv: Path) -> dict:
    """Run Whisper-tiny over a CSV of (audio_path, reference, language).

    Computes WER / CER per language under two conditions:
        - raw audio
        - child-pitch-normalised audio (−4.5 semitones)
    """
    try:
        from tutor.asr_adapt import transcribe  # lazy so offline runs don't fail
        import soundfile as sf  # noqa: F401
    except Exception as e:
        return {"error": f"whisper/soundfile not installed: {e}"}

    rows = list(csv.DictReader(audio_csv.open()))
    by_lang = defaultdict(lambda: {"wer_raw": [], "cer_raw": [],
                                    "wer_norm": [], "cer_norm": []})
    import soundfile as sf
    errors = []
    for r in rows:
        try:
            audio, sr = sf.read(r["audio_path"], dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                ratio = 16000 / sr
                n = int(len(audio) * ratio)
                idx = np.linspace(0, len(audio) - 1, n)
                li = np.floor(idx).astype(int)
                ri = np.clip(li + 1, 0, len(audio) - 1)
                audio = audio[li] * (1 - (idx - li)) + audio[ri] * (idx - li)
            # Raw
            tx_raw, _, _ = transcribe(audio, lang_hint=r["language"], child_pitch_correction=False)
            # Normalised
            tx_norm, _, _ = transcribe(audio, lang_hint=r["language"], child_pitch_correction=True)
            ref = r["reference"]
            lang = r["language"]
            by_lang[lang]["wer_raw"].append(wer(ref, tx_raw))
            by_lang[lang]["cer_raw"].append(cer(ref, tx_raw))
            by_lang[lang]["wer_norm"].append(wer(ref, tx_norm))
            by_lang[lang]["cer_norm"].append(cer(ref, tx_norm))
        except Exception as e:
            errors.append({"row": r.get("audio_path"), "error": str(e)})

    out = {}
    for lang, v in by_lang.items():
        out[lang] = {
            "n": len(v["wer_raw"]),
            "wer_raw_mean": round(float(np.mean(v["wer_raw"])), 4) if v["wer_raw"] else None,
            "cer_raw_mean": round(float(np.mean(v["cer_raw"])), 4) if v["cer_raw"] else None,
            "wer_pitchnorm_mean": round(float(np.mean(v["wer_norm"])), 4) if v["wer_norm"] else None,
            "cer_pitchnorm_mean": round(float(np.mean(v["cer_norm"])), 4) if v["cer_norm"] else None,
            "wer_improvement_abs": round(
                float(np.mean(v["wer_raw"]) - np.mean(v["wer_norm"])), 4
            ) if v["wer_raw"] and v["wer_norm"] else None,
        }
    return {"per_language": out, "errors": errors[:10]}


# ======================================================================
# Report writer
# ======================================================================
def write_markdown(all_stats: dict):
    lines = ["# ASR Pipeline — Reported Metrics\n"]

    p = all_stats["number_word_parser"]
    lines.append("## 1. Number-word parser (extract_integer)\n")
    lines.append(f"- Overall accuracy: **{p['accuracy']*100:.1f}%** "
                 f"({p['correct']}/{p['total']})\n")
    lines.append("| Language | N | Accuracy |")
    lines.append("|---|---|---|")
    for lang, v in p["per_language"].items():
        lines.append(f"| {lang} | {v['n']} | {v['accuracy']*100:.1f}% |")
    lines.append("")

    l = all_stats["language_detection"]
    lines.append("## 2. Language detection (lang_detect.detect)\n")
    lines.append(f"- Overall accuracy: **{l['accuracy']*100:.1f}%** "
                 f"on {l['total']} utterances (30 per class)\n")
    lines.append("| Language | Support | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|")
    for lang, v in l["per_language"].items():
        lines.append(f"| {lang} | {v['support']} | {v['precision']:.3f} | "
                     f"{v['recall']:.3f} | {v['f1']:.3f} |")
    lines.append("")

    ps = all_stats["pitch_shift"]
    lines.append("## 3. Pitch-shift normaliser stability\n")
    lines.append(f"- Semitone shift applied at inference: `{ps['semitones']}` "
                 f"(child → adult register)\n")
    lines.append(f"- Energy ratio (output / input) across 0.2–2.0 s sines: "
                 f"**{ps['mean_energy_ratio']:.3f}** (target ≈ 1.0)\n")
    lines.append(f"- NaN / Inf stable across all durations: **{ps['all_stable']}**\n")

    if "whisper_wer" in all_stats:
        w = all_stats["whisper_wer"]
        lines.append("## 4. Whisper-tiny WER on real audio\n")
        if "error" in w:
            lines.append(f"- _Not run_ — {w['error']}\n")
        else:
            lines.append("| Language | N | WER (raw) | WER (pitch-norm) | Δ WER | CER (raw) |")
            lines.append("|---|---|---|---|---|---|")
            for lang, v in w["per_language"].items():
                lines.append(f"| {lang} | {v['n']} | {v['wer_raw_mean']} | "
                             f"{v['wer_pitchnorm_mean']} | {v['wer_improvement_abs']} | "
                             f"{v['cer_raw_mean']} |")
            lines.append("")
    else:
        lines.append("## 4. Whisper-tiny WER on real audio\n")
        lines.append("_Run with_ `python3 scripts/eval_asr.py --audio-csv <your.csv>`\n"
                     "where the CSV has columns `audio_path,reference,language`. "
                     "Recommended eval set: Mozilla Common Voice Kinyarwanda "
                     "v17 test split filtered to numeric utterances.\n")

    out = FIG_DIR / "asr_metrics.md"
    out.write_text("\n".join(lines))
    print(f"[md] wrote {out}")


def plot_confusion(stats: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cm = stats["language_detection"]["confusion_matrix"]
    langs = list(cm.keys())
    grid = np.array([[cm[g].get(p, 0) for p in langs] for g in langs], dtype=int)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(grid, cmap="Blues")
    ax.set_xticks(range(len(langs)))
    ax.set_yticks(range(len(langs)))
    ax.set_xticklabels([f"pred:{l}" for l in langs])
    ax.set_yticklabels([f"gold:{l}" for l in langs])
    for i in range(len(langs)):
        for j in range(len(langs)):
            ax.text(j, i, str(grid[i, j]), ha="center", va="center",
                    color="white" if grid[i, j] > grid.max() / 2 else "black",
                    fontsize=12, fontweight="bold")
    ax.set_title("Language Detection — Confusion Matrix")
    plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "asr_confusion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] wrote {FIG_DIR/'asr_confusion.png'}")


# ======================================================================
# Main
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-csv", type=Path, default=None,
                    help="Optional CSV of audio_path,reference,language for WER")
    args = ap.parse_args()

    print("=== 1. Number-word parser ===")
    p = eval_number_word_parser()
    print(f"  accuracy: {p['accuracy']*100:.1f}% ({p['correct']}/{p['total']})")
    for lang, v in p["per_language"].items():
        print(f"    {lang}: {v['accuracy']*100:.1f}% ({v['correct']}/{v['n']})")

    print("\n=== 2. Language detection ===")
    l = eval_language_detection()
    print(f"  accuracy: {l['accuracy']*100:.1f}% on {l['total']} utterances")
    for lang, v in l["per_language"].items():
        print(f"    {lang}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} (n={v['support']})")

    print("\n=== 3. Pitch-shift stability ===")
    ps = eval_pitch_shift()
    print(f"  mean energy ratio: {ps['mean_energy_ratio']:.3f}   "
          f"all stable (no NaN/Inf): {ps['all_stable']}")

    all_stats = {
        "number_word_parser": p,
        "language_detection": l,
        "pitch_shift": ps,
    }

    if args.audio_csv:
        print(f"\n=== 4. Whisper WER on {args.audio_csv} ===")
        all_stats["whisper_wer"] = eval_whisper_wer(args.audio_csv)

    out_json = FIG_DIR / "asr_metrics.json"
    out_json.write_text(json.dumps(all_stats, indent=2, ensure_ascii=False))
    print(f"\n[json] wrote {out_json}")
    write_markdown(all_stats)
    plot_confusion(all_stats)


if __name__ == "__main__":
    main()
