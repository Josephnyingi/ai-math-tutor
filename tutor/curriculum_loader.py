"""
Loads curriculum items from JSON, filters by skill / difficulty / age band,
and provides batched access for the adaptive engine.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

SKILLS = ["counting", "number_sense", "addition", "subtraction", "word_problem"]

Item = Dict[str, Any]


def load(path: str | Path) -> List[Item]:
    """Return all curriculum items from *path* (JSON array)."""
    with open(path, "r", encoding="utf-8") as fh:
        items = json.load(fh)
    for item in items:
        if item.get("skill") not in SKILLS:
            raise ValueError(f"Unknown skill in item {item['id']}: {item['skill']}")
    return items


def filter_items(
    items: List[Item],
    skill: Optional[str] = None,
    min_diff: int = 1,
    max_diff: int = 10,
    age_band: Optional[str] = None,
) -> List[Item]:
    out = []
    for it in items:
        if skill and it["skill"] != skill:
            continue
        d = it.get("difficulty", 5)
        if not (min_diff <= d <= max_diff):
            continue
        if age_band and it.get("age_band") and it["age_band"] != age_band:
            continue
        out.append(it)
    return out


def get_by_id(items: List[Item], item_id: str) -> Optional[Item]:
    for it in items:
        if it["id"] == item_id:
            return it
    return None


def sample_diagnostic_probes(
    items: List[Item],
    n_per_skill: int = 1,
    diff_min: int = 1,
    diff_max: int = 10,
) -> List[Item]:
    """Return one probe per skill at age-appropriate difficulty, lowest first."""
    probes: List[Item] = []
    for skill in SKILLS:
        candidates = sorted(
            [
                it for it in items
                if it["skill"] == skill
                and diff_min <= it.get("difficulty", 5) <= diff_max
            ],
            key=lambda x: x.get("difficulty", 5),
        )
        if not candidates:
            # fallback: any item for this skill
            candidates = sorted(
                [it for it in items if it["skill"] == skill],
                key=lambda x: x.get("difficulty", 5),
            )
        probes.extend(candidates[:n_per_skill])
    random.shuffle(probes)
    return probes


def stem(item: Item, lang: str = "en") -> str:
    """Return the question stem in the requested language, fall back to EN."""
    key = f"stem_{lang}"
    return item.get(key) or item.get("stem_en", "")


def tts_path(item: Item, lang: str = "en") -> Optional[str]:
    key = f"tts_{lang}"
    return item.get(key)
