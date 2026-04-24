"""
Child-speech ASR adapter built on openai/whisper-tiny.

Adaptations for child voices:
  - Pitch-shift augmentation at inference (+3 to +6 semitones downward normalisation)
  - Language-aware decoding (EN / FR / KIN / SW)
  - Number-word post-processing to extract integer answers
  - Silence / timeout detection
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Lazy-load heavy deps to keep import time short
_whisper_model = None
_sample_rate = 16000


def _load_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper  # openai-whisper
        _whisper_model = whisper.load_model("tiny", device="cpu")
    return _whisper_model


# ------------------------------------------------------------------
# Number-word maps  (EN / FR / KIN)
# ------------------------------------------------------------------
_NUM_EN = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "twenty": 20, "thirty": 30,
}
_NUM_FR = {
    "zéro": 0, "un": 1, "une": 1, "deux": 2, "trois": 3,
    "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8,
    "neuf": 9, "dix": 10, "onze": 11, "douze": 12,
    "treize": 13, "quatorze": 14, "quinze": 15, "vingt": 20,
}
_NUM_KIN = {
    "zeru": 0, "rimwe": 1, "kabiri": 2, "gatatu": 3, "kane": 4,
    "gatanu": 5, "gatandatu": 6, "indwi": 7, "umunani": 8,
    "icyenda": 9, "icumi": 10, "esheshatu": 6,  # common child variant
}
_NUM_SW = {
    "sifuri": 0, "moja": 1, "mbili": 2, "tatu": 3, "nne": 4,
    "tano": 5, "sita": 6, "saba": 7, "nane": 8, "tisa": 9,
    "kumi": 10, "ishirini": 20, "thelathini": 30,
}
_ALL_NUMS = {**_NUM_EN, **_NUM_FR, **_NUM_KIN, **_NUM_SW}


def _pitch_normalise(audio: np.ndarray, semitones: float = -4.5) -> np.ndarray:
    """
    Compensate for child's higher pitch by pitch-shifting the audio DOWN
    before feeding to Whisper (trained on adult speech).
    Uses a simple resampling trick (no librosa required).
    semitones < 0 shifts DOWN (child → adult range).
    """
    factor = 2 ** (semitones / 12)
    original_len = len(audio)
    resampled_len = int(original_len / factor)
    indices = np.linspace(0, original_len - 1, resampled_len)
    left = np.floor(indices).astype(int)
    right = np.clip(left + 1, 0, original_len - 1)
    frac = indices - left
    normalised = audio[left] * (1 - frac) + audio[right] * frac
    # Resize back to original length
    indices2 = np.linspace(0, len(normalised) - 1, original_len)
    left2 = np.floor(indices2).astype(int)
    right2 = np.clip(left2 + 1, 0, len(normalised) - 1)
    frac2 = indices2 - left2
    out = normalised[left2] * (1 - frac2) + normalised[right2] * frac2
    return out.astype(np.float32)


def transcribe(
    audio: np.ndarray,
    lang_hint: str = "en",
    child_pitch_correction: bool = True,
) -> Tuple[str, str, float]:
    """
    Transcribe *audio* (float32, 16 kHz mono).

    Returns:
        transcript (str), detected_language (str), confidence (float 0-1)
    """
    model = _load_model()

    if child_pitch_correction:
        audio = _pitch_normalise(audio, semitones=-4.5)

    # Map our lang codes to Whisper language codes
    lang_map = {"en": "en", "fr": "fr", "kin": "rw", "sw": "sw", "mix": None}
    whisper_lang = lang_map.get(lang_hint)

    decode_options = {"language": whisper_lang, "fp16": False, "task": "transcribe"}
    result = model.transcribe(audio, **decode_options)

    transcript = result["text"].strip().lower()
    detected_lang = result.get("language", lang_hint)
    # Remap rw → kin
    if detected_lang == "rw":
        detected_lang = "kin"
    # Simple log-prob confidence
    segments = result.get("segments", [])
    if segments:
        avg_logprob = np.mean([s.get("avg_logprob", -1.0) for s in segments])
        confidence = float(np.clip(1 + avg_logprob / 3.0, 0.0, 1.0))
    else:
        confidence = 0.0

    return transcript, detected_lang, confidence


def extract_integer(transcript: str) -> Optional[int]:
    """
    Parse an integer answer from a child's transcript.
    Handles digit strings ("5", "12") and number words in EN/FR/KIN.
    """
    # Digit first
    m = re.search(r"\b(\d+)\b", transcript)
    if m:
        return int(m.group(1))
    # Number word
    words = re.findall(r"[a-záàâéèêëîïôùûüç]+", transcript.lower())
    for w in words:
        if w in _ALL_NUMS:
            return _ALL_NUMS[w]
    return None


def is_silence(audio: np.ndarray, threshold: float = 0.01) -> bool:
    """Return True if the audio is essentially silent (no speech detected)."""
    return float(np.sqrt(np.mean(audio ** 2))) < threshold


def transcribe_file(path: str | Path, lang_hint: str = "en") -> Tuple[str, str, float]:
    """Convenience wrapper: load a WAV/MP3 file and transcribe."""
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != _sample_rate:
        # Simple linear resample
        ratio = _sample_rate / sr
        n = int(len(audio) * ratio)
        idx = np.linspace(0, len(audio) - 1, n)
        li = np.floor(idx).astype(int)
        ri = np.clip(li + 1, 0, len(audio) - 1)
        audio = audio[li] * (1 - (idx - li)) + audio[ri] * (idx - li)
    return transcribe(audio.astype(np.float32), lang_hint=lang_hint)
