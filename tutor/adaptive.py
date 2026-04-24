"""
Adaptive engine: Bayesian Knowledge Tracing (BKT) + Elo baseline.

BKT parameters per skill:
  p_learn  : probability of learning after each attempt
  p_guess  : probability of correct response despite not knowing
  p_slip   : probability of incorrect response despite knowing
  p_known  : current belief learner already knows the skill (updated each response)

Elo: skill rating per sub-skill updated via standard K-factor after each response.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SKILLS = ["counting", "number_sense", "addition", "subtraction", "word_problem"]

# Default BKT priors (can be overridden per skill)
DEFAULT_BKT = {
    "p_learn": 0.20,
    "p_guess": 0.25,
    "p_slip": 0.10,
    "p_known": 0.10,
}

ELO_K = 32
ELO_INIT = 800
ITEM_ELO_INIT = 1000  # items start slightly harder than learners


@dataclass
class BKTSkillState:
    p_known: float = DEFAULT_BKT["p_known"]
    p_learn: float = DEFAULT_BKT["p_learn"]
    p_guess: float = DEFAULT_BKT["p_guess"]
    p_slip: float = DEFAULT_BKT["p_slip"]
    attempts: int = 0
    correct: int = 0

    def update(self, is_correct: bool) -> None:
        """Standard BKT posterior update."""
        pk = self.p_known
        if is_correct:
            numerator = pk * (1 - self.p_slip)
            denominator = numerator + (1 - pk) * self.p_guess
        else:
            numerator = pk * self.p_slip
            denominator = numerator + (1 - pk) * (1 - self.p_guess)
        pk_given_obs = numerator / (denominator + 1e-9)
        # Learning transition
        self.p_known = pk_given_obs + (1 - pk_given_obs) * self.p_learn
        self.attempts += 1
        if is_correct:
            self.correct += 1

    @property
    def mastery(self) -> float:
        """Mastery probability in [0, 1]."""
        return self.p_known

    def predict_correct(self) -> float:
        """Expected P(correct) for next item."""
        return self.p_known * (1 - self.p_slip) + (1 - self.p_known) * self.p_guess


@dataclass
class EloSkillState:
    rating: float = ELO_INIT

    def update(self, item_difficulty: int, is_correct: bool) -> None:
        item_rating = ELO_INIT + (item_difficulty - 5) * 50
        expected = 1.0 / (1 + 10 ** ((item_rating - self.rating) / 400))
        self.rating += ELO_K * (int(is_correct) - expected)

    def predict_correct(self, item_difficulty: int) -> float:
        item_rating = ELO_INIT + (item_difficulty - 5) * 50
        return 1.0 / (1 + 10 ** ((item_rating - self.rating) / 400))

    @property
    def mastery(self) -> float:
        """Normalise Elo rating to [0,1] range for reporting."""
        return max(0.0, min(1.0, (self.rating - 400) / 1200))


@dataclass
class LearnerState:
    learner_id: str
    lang: str = "en"
    bkt: Dict[str, BKTSkillState] = field(default_factory=dict)
    elo: Dict[str, EloSkillState] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    session_count: int = 0
    plateau_sessions: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        for skill in SKILLS:
            if skill not in self.bkt:
                self.bkt[skill] = BKTSkillState()
            if skill not in self.elo:
                self.elo[skill] = EloSkillState()
            if skill not in self.plateau_sessions:
                self.plateau_sessions[skill] = 0

    def record_response(self, item: dict, is_correct: bool) -> None:
        skill = item["skill"]
        diff = item.get("difficulty", 5)
        prev_mastery = self.bkt[skill].mastery
        self.bkt[skill].update(is_correct)
        self.elo[skill].update(diff, is_correct)
        self.history.append({
            "item_id": item["id"],
            "skill": skill,
            "difficulty": diff,
            "correct": is_correct,
            "bkt_mastery_after": self.bkt[skill].mastery,
        })
        # Plateau detection: mastery didn't improve despite low difficulty
        if diff <= 3 and (self.bkt[skill].mastery - prev_mastery) < 0.02:
            self.plateau_sessions[skill] = self.plateau_sessions.get(skill, 0) + 1
        else:
            self.plateau_sessions[skill] = 0

    def dyscalculia_warning(self) -> List[str]:
        """Skills plateaued for 3+ sessions despite easy items."""
        return [s for s, n in self.plateau_sessions.items() if n >= 3]

    def select_next_item(self, items: list, use_bkt: bool = True) -> Optional[dict]:
        """
        Choose the next item targeting the skill with lowest mastery,
        at a difficulty one step above the learner's current Elo estimate.
        BKT mode: use p_known; Elo mode: use normalised rating.
        """
        if not items:
            return None

        # Target weakest skill
        if use_bkt:
            weakest = min(SKILLS, key=lambda s: self.bkt[s].mastery)
        else:
            weakest = min(SKILLS, key=lambda s: self.elo[s].mastery)

        # Estimate learner's current difficulty sweet-spot (zone of proximal dev)
        if use_bkt:
            mastery = self.bkt[weakest].mastery
        else:
            mastery = self.elo[weakest].mastery
        target_diff = max(1, min(10, int(mastery * 10) + 1))

        candidates = [
            it for it in items
            if it["skill"] == weakest
            and abs(it.get("difficulty", 5) - target_diff) <= 2
        ]
        if not candidates:
            candidates = [it for it in items if it["skill"] == weakest]
        if not candidates:
            candidates = items

        # Prefer items not yet seen
        seen_ids = {h["item_id"] for h in self.history}
        unseen = [it for it in candidates if it["id"] not in seen_ids]
        pool = unseen if unseen else candidates
        # Closest difficulty match
        pool.sort(key=lambda x: abs(x.get("difficulty", 5) - target_diff))
        return pool[0]

    def skill_summary(self) -> Dict[str, Dict]:
        return {
            s: {
                "current": round(self.bkt[s].mastery, 3),
                "delta": round(
                    self.bkt[s].mastery
                    - (self.history[-6]["bkt_mastery_after"]
                       if len(self.history) >= 6 else 0.0),
                    3,
                ),
                "attempts": self.bkt[s].attempts,
            }
            for s in SKILLS
        }

    def to_dict(self) -> dict:
        return {
            "learner_id": self.learner_id,
            "lang": self.lang,
            "session_count": self.session_count,
            "bkt": {s: vars(self.bkt[s]) for s in SKILLS},
            "elo": {s: {"rating": self.elo[s].rating} for s in SKILLS},
            "plateau_sessions": self.plateau_sessions,
            "history": self.history[-100:],  # keep last 100
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LearnerState":
        state = cls(learner_id=d["learner_id"], lang=d.get("lang", "en"))
        state.session_count = d.get("session_count", 0)
        state.history = d.get("history", [])
        state.plateau_sessions = d.get("plateau_sessions", {s: 0 for s in SKILLS})
        for s in SKILLS:
            if s in d.get("bkt", {}):
                state.bkt[s] = BKTSkillState(**d["bkt"][s])
            if s in d.get("elo", {}):
                state.elo[s].rating = d["elo"][s]["rating"]
        return state
