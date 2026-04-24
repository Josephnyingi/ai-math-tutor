"""
Lightweight language detection for child responses (KIN / FR / EN / SW / mix).

Strategy:
  1. Word-level lookup against EN / FR / KIN / SW number-word + common-word lexicons.
  2. If no words match, fall back to character n-gram scoring.
  3. 'mix' is returned when the top-two languages both contribute ≥ 15 % of tokens.

All lookups are purely in-memory — zero network calls.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, Tuple

# ------------------------------------------------------------------
# Lexicons (numbers + very common math-context words per language)
# ------------------------------------------------------------------
_LEX_EN = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "how", "many", "plus", "minus", "equals", "and",
    "the", "is", "are", "yes", "no",
}
_LEX_FR = {
    "zéro", "zero", "un", "une", "deux", "trois", "quatre", "cinq", "six",
    "sept", "huit", "neuf", "dix", "onze", "douze", "treize", "quatorze",
    "quinze", "seize", "vingt", "trente", "quarante", "cinquante",
    "combien", "plus", "moins", "égale", "et", "le", "la", "les", "oui",
    "non", "de",
}
_LEX_KIN = {
    "zeru", "rimwe", "kabiri", "gatatu", "kane", "gatanu", "gatandatu",
    "indwi", "umunani", "icyenda", "icumi", "cumi", "makumyabiri",
    "mirongo", "esheshatu", "twewenti", "angahe", "nka", "ni", "na",
    "yego", "oya", "soma", "bara", "ibara",
}
_LEX_SW = {
    "sifuri", "moja", "mbili", "tatu", "nne", "tano", "sita", "saba",
    "nane", "tisa", "kumi", "ishirini", "thelathini", "arobaini",
    "hamsini", "ngapi", "jumlisha", "toa", "punguza", "sawa", "ndiyo",
    "hapana", "na", "ni", "jibu", "hesabu", "namba",
}

_LEXICONS: Dict[str, set] = {"en": _LEX_EN, "fr": _LEX_FR, "kin": _LEX_KIN, "sw": _LEX_SW}

# Character n-gram profiles (trigram sets common per language)
_NGRAM_EN = {"the", "ing", "ion", "and", "ent", "hat", "tha"}
_NGRAM_FR = {"ent", "les", "des", "une", "ous", "est", "que"}
_NGRAM_KIN = {"nyi", "mwe", "abi", "and", "umi", "iku", "gat"}
_NGRAM_SW = {"aku", "ana", "kwa", "una", "ndi", "ume", "ote"}


def _tokenise(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text.lower())
    return re.findall(r"[a-záàâéèêëîïôùûüçñ]+", text)


def _score_lexicon(tokens: list[str]) -> Dict[str, float]:
    scores: Dict[str, float] = {"en": 0, "fr": 0, "kin": 0, "sw": 0}
    n = max(len(tokens), 1)
    for tok in tokens:
        for lang, lex in _LEXICONS.items():
            if tok in lex:
                scores[lang] += 1
    return {lang: count / n for lang, count in scores.items()}


def _score_ngrams(text: str) -> Dict[str, float]:
    text = text.lower()
    total = max(len(text) - 2, 1)
    trigrams = {text[i: i + 3] for i in range(len(text) - 2)}
    profiles = {"en": _NGRAM_EN, "fr": _NGRAM_FR, "kin": _NGRAM_KIN, "sw": _NGRAM_SW}
    return {lang: len(trigrams & prof) / total for lang, prof in profiles.items()}


def detect(text: str) -> Tuple[str, Dict[str, float]]:
    """
    Detect the language of *text*.

    Returns:
        dominant_lang: "en" | "fr" | "kin" | "mix"
        scores: dict of per-language scores [0, 1]
    """
    tokens = _tokenise(text)
    if not tokens:
        return "en", {"en": 0.0, "fr": 0.0, "kin": 0.0}

    lex_scores = _score_lexicon(tokens)
    ngram_scores = _score_ngrams(text)

    # Combine: lexicon is more reliable, n-gram is tie-breaker
    combined = {
        lang: 0.7 * lex_scores[lang] + 0.3 * ngram_scores[lang]
        for lang in ("en", "fr", "kin", "sw")
    }

    total = sum(combined.values()) or 1.0
    normalised = {lang: v / total for lang, v in combined.items()}

    sorted_langs = sorted(normalised, key=normalised.get, reverse=True)
    top, second = sorted_langs[0], sorted_langs[1]

    # 'mix' when two languages both exceed 15 % share
    if normalised[second] >= 0.15:
        dominant = "mix"
    else:
        dominant = top

    return dominant, normalised


def reply_lang(detected: str, fallback: str = "en") -> str:
    """
    Resolve which language to reply in.
    For 'mix', return the single highest-scoring language.
    """
    if detected in ("en", "fr", "kin", "sw"):
        return detected
    return fallback


def extract_number_words(text: str, lang: str) -> list[str]:
    """Return number words from *text* that belong to *lang*'s lexicon."""
    tokens = _tokenise(text)
    num_words_en = {
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    }
    num_words_fr = {
        "zéro", "un", "une", "deux", "trois", "quatre", "cinq", "six", "sept",
        "huit", "neuf", "dix", "onze", "douze", "treize", "quatorze", "quinze",
        "seize", "vingt",
    }
    num_words_kin = {
        "zeru", "rimwe", "kabiri", "gatatu", "kane", "gatanu", "gatandatu",
        "indwi", "umunani", "icyenda", "icumi",
    }
    num_words_sw = {
        "sifuri", "moja", "mbili", "tatu", "nne", "tano", "sita", "saba",
        "nane", "tisa", "kumi", "ishirini",
    }
    mapping = {"en": num_words_en, "fr": num_words_fr, "kin": num_words_kin, "sw": num_words_sw}
    pool = mapping.get(lang, set())
    return [t for t in tokens if t in pool]
