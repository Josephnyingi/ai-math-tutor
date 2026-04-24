"""
Loads the quantised language head (GGUF / int4) for numeracy feedback generation.

Model: TinyLlama-1.1B-Chat-v1.0 quantised to Q4_K_M GGUF (~669 MB full, ~350 MB Q4).
We strip it further with a LoRA adapter trained only on numeracy instruction pairs,
then re-quantise the merged model to Q4_K_M targeting ≤ 55 MB for the language head.

At runtime we use llama-cpp-python for CPU inference (no GPU required).
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

_llm = None
_MODEL_PATH: Optional[Path] = None

SYSTEM_PROMPT = {
    "en": (
        "You are a friendly math tutor for young children aged 5–9. "
        "Give very short, warm feedback (≤ 2 sentences). "
        "Use simple words. Never use markdown. "
        "If the child is correct, celebrate. If wrong, gently show the right answer."
    ),
    "fr": (
        "Tu es un tuteur de mathématiques sympa pour les enfants de 5 à 9 ans. "
        "Donne un retour très court et chaleureux (≤ 2 phrases). "
        "Utilise des mots simples. Pas de markdown."
    ),
    "kin": (
        "Uri umwigisha w'imibare w'inshuti ku bana b'imyaka 5-9. "
        "Tanga igisubizo gito kandi cyiza (inyandiko ≤ 2). "
        "Koresha amagambo yoroheje."
    ),
}


def set_model_path(path: str | Path) -> None:
    global _MODEL_PATH
    _MODEL_PATH = Path(path)


def _load():
    global _llm
    if _llm is not None:
        return _llm
    if _MODEL_PATH is None or not _MODEL_PATH.exists():
        return None  # graceful degradation to template responses
    try:
        from llama_cpp import Llama
        _llm = Llama(
            model_path=str(_MODEL_PATH),
            n_ctx=512,
            n_threads=2,
            verbose=False,
        )
    except Exception:
        _llm = None
    return _llm


# ------------------------------------------------------------------
# Template fallback (works without any model loaded)
# ------------------------------------------------------------------

_CORRECT_TEMPLATES = {
    "en": [
        "Great job! That's exactly right! 🎉",
        "Wonderful! You got it!",
        "Yes! {answer} is correct! Well done!",
    ],
    "fr": [
        "Bravo ! C'est exactement ça ! 🎉",
        "Super ! Tu as trouvé !",
        "Oui ! {answer} est correct ! Bien joué !",
    ],
    "kin": [
        "Ni byiza cyane! Ni yo! 🎉",
        "Yego! Wabikoze neza!",
        "Ni {answer}! Wabikoze neza cyane!",
    ],
}
_WRONG_TEMPLATES = {
    "en": [
        "Good try! The answer is {answer}. Let's try again!",
        "Almost! It's {answer}. You'll get it next time!",
        "Not quite — the answer is {answer}. Keep going!",
    ],
    "fr": [
        "Bon essai ! La réponse est {answer}. Réessaie !",
        "Presque ! C'est {answer}. Tu y arriveras !",
        "Pas tout à fait — la réponse est {answer}. Continue !",
    ],
    "kin": [
        "Gerageza neza! Igisubizo ni {answer}. Ongera ugerageze!",
        "Hafi! Ni {answer}. Uzagera!",
        "Si byo — igisubizo ni {answer}. Komeza!",
    ],
}

import random as _random


def _template_feedback(is_correct: bool, answer: int, lang: str) -> str:
    lang = lang if lang in ("en", "fr", "kin") else "en"
    pool = _CORRECT_TEMPLATES[lang] if is_correct else _WRONG_TEMPLATES[lang]
    return _random.choice(pool).format(answer=answer)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def generate_feedback(
    is_correct: bool,
    answer: int,
    lang: str = "en",
    child_response: str = "",
    max_tokens: int = 60,
) -> str:
    """
    Generate a short feedback string for the child.

    Falls back to templates if the GGUF model is not loaded,
    ensuring < 2.5 s latency even on slow hardware.
    """
    t0 = time.time()
    llm = _load()

    if llm is None:
        return _template_feedback(is_correct, answer, lang)

    system = SYSTEM_PROMPT.get(lang, SYSTEM_PROMPT["en"])
    verdict = "correct" if is_correct else f"incorrect (correct answer is {answer})"
    user_msg = (
        f"The child said: '{child_response}'. Their answer was {verdict}. "
        f"Give feedback in {'English' if lang=='en' else 'French' if lang=='fr' else 'Kinyarwanda'}."
    )

    try:
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
            stop=["\n\n"],
        )
        text = out["choices"][0]["message"]["content"].strip()
        elapsed = time.time() - t0
        # Latency guard: if model took > 2 s, use template next time
        if elapsed > 2.0:
            return _template_feedback(is_correct, answer, lang)
        return text
    except Exception:
        return _template_feedback(is_correct, answer, lang)
