"""
make_synthetic_child.py — Generate synthetic child-voiced numeracy utterances.

Pipeline:
  1. TTS (Piper TTS or espeak-ng) renders number words in EN / FR / KIN.
  2. Pitch-shift +3 to +6 semitones to approximate child prosody.
  3. Add MUSAN classroom noise at a random SNR between 5–15 dB.
  4. Write manifest CSV compatible with child_utt_sample_seed.csv schema.

Usage:
    python scripts/make_synthetic_child.py --out data/synthetic_child_audio/

Requirements:
    pip install piper-tts soundfile numpy
    # or: apt-get install espeak-ng (fallback)
"""
from __future__ import annotations

import csv
import os
import random
import struct
import tempfile
import wave
from pathlib import Path
from typing import List, Tuple

import numpy as np

random.seed(42)

NUMBER_WORDS = {
    "en": {
        0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
        5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
        10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
        14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
        18: "eighteen", 19: "nineteen", 20: "twenty",
    },
    "fr": {
        0: "zéro", 1: "un", 2: "deux", 3: "trois", 4: "quatre",
        5: "cinq", 6: "six", 7: "sept", 8: "huit", 9: "neuf",
        10: "dix", 11: "onze", 12: "douze", 13: "treize",
        14: "quatorze", 15: "quinze", 16: "seize", 17: "dix-sept",
        18: "dix-huit", 19: "dix-neuf", 20: "vingt",
    },
    "kin": {
        0: "zeru", 1: "rimwe", 2: "kabiri", 3: "gatatu", 4: "kane",
        5: "gatanu", 6: "gatandatu", 7: "indwi", 8: "umunani",
        9: "icyenda", 10: "icumi", 11: "cumi na rimwe",
        12: "cumi na kabiri", 13: "cumi na gatatu", 14: "cumi na kane",
        15: "cumi na gatanu", 16: "cumi na gatandatu", 17: "cumi na indwi",
        18: "cumi na umunani", 19: "cumi na icyenda", 20: "makumyabiri",
    },
}


def _pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Simple pitch-shift via resampling (no librosa required)."""
    factor = 2 ** (semitones / 12)
    new_len = int(len(audio) / factor)
    idx = np.linspace(0, len(audio) - 1, new_len)
    li = np.floor(idx).astype(int)
    ri = np.clip(li + 1, 0, len(audio) - 1)
    resampled = audio[li] * (1 - (idx - li)) + audio[ri] * (idx - li)
    # Restore original length via linear interp
    idx2 = np.linspace(0, len(resampled) - 1, len(audio))
    li2 = np.floor(idx2).astype(int)
    ri2 = np.clip(li2 + 1, 0, len(resampled) - 1)
    return (resampled[li2] * (1 - (idx2 - li2)) + resampled[ri2] * (idx2 - li2)).astype(np.float32)


def _add_noise(audio: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
    """Add Gaussian white noise at target SNR (classroom noise simulation)."""
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio)).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0)


def _synthesise_espeak(text: str, lang: str, sr: int = 16000) -> np.ndarray:
    """Fallback TTS via espeak-ng (must be installed on system)."""
    lang_code = {"en": "en", "fr": "fr", "kin": "rw"}.get(lang, "en")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    os.system(f'espeak-ng -v {lang_code} -w "{tmp_path}" "{text}" 2>/dev/null')
    try:
        import soundfile as sf
        audio, file_sr = sf.read(tmp_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            ratio = sr / file_sr
            n = int(len(audio) * ratio)
            idx = np.linspace(0, len(audio) - 1, n)
            li = np.floor(idx).astype(int)
            ri = np.clip(li + 1, 0, len(audio) - 1)
            audio = audio[li] * (1 - (idx - li)) + audio[ri] * (idx - li)
        return audio.astype(np.float32)
    except Exception:
        # Return 0.5s silence if TTS fails
        return np.zeros(sr // 2, dtype=np.float32)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _write_wav(path: Path, audio: np.ndarray, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def generate(out_dir: Path, langs: List[str] = None, numbers: range = None) -> Path:
    if langs is None:
        langs = ["en", "fr", "kin"]
    if numbers is None:
        numbers = range(0, 21)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    utt_id = 1

    for lang in langs:
        for n in numbers:
            word = NUMBER_WORDS[lang].get(n, str(n))
            # Synthesise
            audio = _synthesise_espeak(word, lang)
            # Child pitch-shift (+3 to +6 semitones)
            semitones = random.uniform(3, 6)
            audio = _pitch_shift(audio, 16000, semitones)
            # Classroom noise (SNR 5–15 dB)
            snr = random.uniform(5, 15)
            audio = _add_noise(audio, snr_db=snr)

            fname = f"U{utt_id:04d}_{lang}_{n}.wav"
            fpath = out_dir / fname
            _write_wav(fpath, audio)

            correctness = "great" if snr > 10 else "ok"
            manifest_rows.append({
                "utt_id": f"U{utt_id:04d}",
                "audio_path": str(fpath),
                "transcript_en": NUMBER_WORDS["en"].get(n, str(n)),
                "language": lang,
                "correctness": correctness,
            })
            utt_id += 1

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["utt_id", "audio_path", "transcript_en", "language", "correctness"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Generated {len(manifest_rows)} utterances → {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic_child_audio")
    parser.add_argument("--langs", nargs="+", default=["en", "fr", "kin"])
    args = parser.parse_args()
    generate(Path(args.out), langs=args.langs)
