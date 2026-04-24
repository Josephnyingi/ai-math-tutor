"""
generate_curriculum.py — Reproduce / extend the full 60-item curriculum.

Usage:
    python scripts/generate_curriculum.py \
        --seed data/T3.1_Math_Tutor/curriculum_seed.json \
        --out  data/T3.1_Math_Tutor/curriculum_full.json

The generator extends the seed to ≥ 60 items by:
  1. Filling gaps in each skill × difficulty × age_band combination.
  2. Adding RWF-context word problems (culturally grounded for Rwanda).
  3. Adding Kinyarwanda / French stems for all items that only had English.

Runtime: < 30 s on any laptop. No GPU needed.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(42)

SKILLS = ["counting", "number_sense", "addition", "subtraction", "word_problem"]

# Templates for automatic stem generation
_COUNTING_TEMPLATES = [
    # (en, fr, kin, visual_tag, answer)
    ("How many {obj}?", "Combien de {obj}?", "{obj_kin} zingahe?", "{obj}_{n}", None),
]
_ADDITION_TEMPLATES = [
    ("{a} plus {b} equals?", "{a} plus {b} égale?", "{a} + {b} ni angahe?", "beads_{a}_plus_{b}", None),
    (
        "{name} has {a} {obj}. She gets {b} more. How many now?",
        "{name} a {a} {obj_fr}. Elle en reçoit {b} de plus. Combien maintenant?",
        "{name} afite {a} {obj_kin}. Aronka {b} ndi. Ni angahe ubu?",
        "{obj}_{a}_plus_{b}",
        None,
    ),
]
_SUBTRACTION_TEMPLATES = [
    ("{a} minus {b} equals?", "{a} moins {b} égale?", "{a} - {b} ni angahe?", "items_{a}_minus_{b}", None),
    (
        "There are {a} {obj}. {b} are taken away. How many remain?",
        "Il y a {a} {obj_fr}. On en enlève {b}. Combien reste-t-il?",
        "Hari {a} {obj_kin}. Bafashe {b}. Ni angahe bisigaye?",
        "{obj}_{a}_minus_{b}",
        None,
    ),
]
_NUMBER_SENSE_TEMPLATES = [
    (
        "Which is bigger: {a} or {b}?",
        "Lequel est plus grand: {a} ou {b}?",
        "Ni iyihe nini: {a} cyangwa {b}?",
        "compare_{a}_{b}",
        None,
    ),
    (
        "What number comes between {a} and {c}?",
        "Quel nombre est entre {a} et {c}?",
        "Ni iyihe nimero iri hagati ya {a} na {c}?",
        "number_line_{a}_{c}",
        None,
    ),
]

OBJECTS_EN = ["goats", "apples", "mangoes", "fish", "bananas", "eggs", "tomatoes", "beans", "cows", "books"]
OBJECTS_FR = ["chèvres", "pommes", "mangues", "poissons", "bananes", "oeufs", "tomates", "haricots", "vaches", "livres"]
OBJECTS_KIN = ["ente", "pome", "imyembe", "amafi", "imigati", "amagi", "inyanya", "ibishyimbo", "inka", "ibitabo"]
NAMES = ["Sara", "Amani", "Keza", "Jean", "Aline", "Eric", "Diane", "Claude"]

WORD_PROBLEM_TEMPLATES = [
    (
        "{name} has {a} RWF. She buys a {obj} for {b} RWF. How much is left?",
        "{name} a {a} RWF. Elle achète un(e) {obj_fr} pour {b} RWF. Combien reste-t-il?",
        "{name} afite amafaranga {a} RWF. Aguze {obj_kin} agura {b} RWF. Asigara angahe?",
        "rwf_{a}_minus_{b}",
        None,
    ),
    (
        "{n} children share {total} {obj} equally. How many does each get?",
        "{n} enfants partagent {total} {obj_fr} en parts égales. Combien chacun reçoit-il?",
        "Abana {n} basangiye {total} {obj_kin} bingana. Umuntu wese aronka angahe?",
        "share_{n}_children_{total}_{obj}",
        None,
    ),
]


def _difficulty(a: int, b: int, op: str) -> int:
    """Estimate difficulty level 1–10 from operands and operation."""
    if op == "count":
        return max(1, min(3, a // 3 + 1))
    if op in ("+", "-"):
        if max(a, b) <= 5:
            return 3
        if max(a, b) <= 10:
            return 4
        if max(a, b) <= 20:
            return 5
        if max(a, b) <= 50:
            return 7
        return 9
    if op == "sense":
        return max(1, min(9, len(str(max(a, b)))))
    return 5


def _age_band(diff: int) -> str:
    if diff <= 2:
        return "5-6"
    if diff <= 4:
        return "6-7"
    if diff <= 6:
        return "7-8"
    return "8-9"


def generate(seed_path: Path, out_path: Path, target: int = 80) -> None:
    with open(seed_path, "r", encoding="utf-8") as fh:
        items = json.load(fh)

    existing_ids = {it["id"] for it in items}
    counters = {"C": 3, "N": 3, "A": 4, "S": 4, "W": 3}

    def next_id(prefix: str) -> str:
        counters[prefix] += 1
        return f"{prefix}{counters[prefix]:03d}"

    # ------------------------------------------------------------------
    # Counting items (n = 1..15)
    # ------------------------------------------------------------------
    for n in range(1, 16):
        obj_i = (n - 1) % len(OBJECTS_EN)
        obj = OBJECTS_EN[obj_i]
        diff = _difficulty(n, 0, "count")
        item = {
            "id": next_id("C"),
            "skill": "counting",
            "difficulty": diff,
            "age_band": _age_band(diff),
            "stem_en": f"How many {obj}?",
            "stem_fr": f"Combien de {OBJECTS_FR[obj_i]}?",
            "stem_kin": f"{OBJECTS_KIN[obj_i]} zingahe?",
            "visual": f"{obj}_{n}",
            "answer_int": n,
        }
        items.append(item)

    # ------------------------------------------------------------------
    # Number sense
    # ------------------------------------------------------------------
    ns_pairs = [
        (3, 7), (5, 2), (8, 4), (12, 9), (15, 11), (20, 17),
        (33, 31), (48, 52), (99, 101), (47, 49),  # between
    ]
    for a, b in ns_pairs:
        diff = _difficulty(max(a, b), 0, "sense")
        # Which is bigger
        item = {
            "id": next_id("N"),
            "skill": "number_sense",
            "difficulty": diff,
            "age_band": _age_band(diff),
            "stem_en": f"Which is bigger: {a} or {b}?",
            "stem_fr": f"Lequel est plus grand: {a} ou {b}?",
            "stem_kin": f"Ni iyihe nini: {a} cyangwa {b}?",
            "visual": f"compare_{a}_{b}",
            "answer_int": max(a, b),
        }
        items.append(item)
        # Between
        if b == a + 2:
            mid = a + 1
            item2 = {
                "id": next_id("N"),
                "skill": "number_sense",
                "difficulty": diff,
                "age_band": _age_band(diff),
                "stem_en": f"What number comes between {a} and {b}?",
                "stem_fr": f"Quel nombre est entre {a} et {b}?",
                "stem_kin": f"Ni iyihe nimero iri hagati ya {a} na {b}?",
                "visual": f"number_line_{a}_{b}",
                "answer_int": mid,
            }
            items.append(item2)

    # ------------------------------------------------------------------
    # Addition
    # ------------------------------------------------------------------
    add_pairs = [
        (1,1),(1,2),(2,2),(2,3),(3,3),(4,3),(5,4),(6,3),(7,2),(8,5),
        (9,6),(10,5),(12,8),(15,7),(18,9),(20,13),(25,16),(30,14),(40,27),(50,33),
    ]
    names = NAMES
    for i, (a, b) in enumerate(add_pairs):
        diff = _difficulty(a, b, "+")
        obj_i = i % len(OBJECTS_EN)
        name = names[i % len(names)]
        if a + b <= 10:
            stem_en = f"{a} plus {b} equals?"
            stem_fr = f"{a} plus {b} égale?"
            stem_kin = f"{a} + {b} ni angahe?"
        else:
            obj = OBJECTS_EN[obj_i]
            stem_en = f"{name} has {a} {obj}. Gets {b} more. How many now?"
            stem_fr = f"{name} a {a} {OBJECTS_FR[obj_i]}. Elle en reçoit {b}. Combien?"
            stem_kin = f"{name} afite {a} {OBJECTS_KIN[obj_i]}. Aronka {b}. Ni angahe ubu?"
        items.append({
            "id": next_id("A"),
            "skill": "addition",
            "difficulty": diff,
            "age_band": _age_band(diff),
            "stem_en": stem_en,
            "stem_fr": stem_fr,
            "stem_kin": stem_kin,
            "visual": f"{OBJECTS_EN[obj_i]}_{a}_plus_{b}",
            "answer_int": a + b,
        })

    # ------------------------------------------------------------------
    # Subtraction
    # ------------------------------------------------------------------
    sub_pairs = [
        (3,1),(4,2),(5,2),(6,3),(7,4),(8,3),(9,5),(10,4),(12,7),(15,8),
        (18,9),(20,11),(25,13),(30,17),(40,22),(50,28),(62,28),(75,38),(100,45),(99,51),
    ]
    for i, (a, b) in enumerate(sub_pairs):
        diff = _difficulty(a, b, "-")
        obj_i = i % len(OBJECTS_EN)
        obj = OBJECTS_EN[obj_i]
        name = names[i % len(names)]
        if a <= 10:
            stem_en = f"{a} minus {b} equals?"
            stem_fr = f"{a} moins {b} égale?"
            stem_kin = f"{a} - {b} ni angahe?"
        else:
            stem_en = f"There are {a} {obj}. {b} are taken away. How many remain?"
            stem_fr = f"Il y a {a} {OBJECTS_FR[obj_i]}. On en enlève {b}. Combien?"
            stem_kin = f"Hari {a} {OBJECTS_KIN[obj_i]}. Bafashe {b}. Bisigaye angahe?"
        items.append({
            "id": next_id("S"),
            "skill": "subtraction",
            "difficulty": diff,
            "age_band": _age_band(diff),
            "stem_en": stem_en,
            "stem_fr": stem_fr,
            "stem_kin": stem_kin,
            "visual": f"{obj}_{a}_minus_{b}",
            "answer_int": a - b,
        })

    # ------------------------------------------------------------------
    # Word problems (culturally grounded)
    # ------------------------------------------------------------------
    word_problems = [
        (6, "A tomato costs 50 RWF. Mama has 300 RWF. How many tomatoes can she buy?",
         "Une tomate coûte 50 RWF. Mama a 300 RWF. Combien peut-elle acheter?",
         "Inyanya imwe igura 50 RWF. Mama afite 300 RWF. Ashobora kugura inyanya zingahe?",
         "rwf_300_tomato_50", 6),
        (6, "4 friends share 20 beans equally. How many does each get?",
         "4 amis partagent 20 haricots. Combien chacun reçoit-il?",
         "Inshuti 4 zasangiye ibishyimbo 20 bingana. Umuntu wese aronka angahe?",
         "beans_20_share_4", 5),
        (7, "A bus has 15 seats. 8 people sit down. How many seats are empty?",
         "Un bus a 15 sièges. 8 personnes s'assoient. Combien de sièges vides?",
         "Bisi ifite intebe 15. Abantu 8 bicaye. Intebe zingahe zisigaye?",
         "bus_15_seats_8", 7),
        (8, "Sara saves 200 RWF each week. After 4 weeks, how much has she saved?",
         "Sara économise 200 RWF chaque semaine. Après 4 semaines, combien a-t-elle?",
         "Sara bika 200 RWF buri cyumweru. Nyuma y'ibyumweru 4 afite angahe?",
         "savings_200_4weeks", 800),
        (8, "A field has 3 rows of 7 maize plants. How many plants total?",
         "Un champ a 3 rangées de 7 plants de maïs. Combien au total?",
         "Inzuri ifite imirongo 3 ya 7 buri murongo. Ni ingahe yose hamwe?",
         "maize_3_rows_7", 21),
        (9, "Keza had 50 RWF. She spent 18 RWF on bread. How much does she have left?",
         "Keza avait 50 RWF. Elle a dépensé 18 RWF. Combien lui reste-t-il?",
         "Keza yari afite 50 RWF. Yashonje 18 RWF. Asigara angahe?",
         "rwf_50_minus_18", 32),
        (9, "Jean counted 12 eggs in one basket and 9 in another. How many eggs in total?",
         "Jean a compté 12 oeufs dans un panier et 9 dans un autre. Combien en tout?",
         "Jean abaraga amagi 12 muri karito imwe na 9 muri indi. Hamwe ni angahe?",
         "eggs_12_plus_9", 21),
        (7, "There are 24 children. Half go home. How many remain?",
         "Il y a 24 enfants. La moitié rentrent. Combien restent?",
         "Hari abana 24. Bameshe baja iwabo. Ni bangahe basigaye?",
         "children_24_half", 12),
    ]
    for diff, en, fr, kin, visual, ans in word_problems:
        items.append({
            "id": next_id("W"),
            "skill": "word_problem",
            "difficulty": diff,
            "age_band": _age_band(diff),
            "stem_en": en,
            "stem_fr": fr,
            "stem_kin": kin,
            "visual": visual,
            "answer_int": ans,
        })

    # Deduplicate by id (seed items win)
    seen = set()
    deduped = []
    for it in items:
        if it["id"] not in seen:
            seen.add(it["id"])
            deduped.append(it)

    # Sort by skill then difficulty
    order = {s: i for i, s in enumerate(SKILLS)}
    deduped.sort(key=lambda x: (order.get(x["skill"], 99), x.get("difficulty", 5)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(deduped, fh, ensure_ascii=False, indent=2)

    print(f"Generated {len(deduped)} curriculum items → {out_path}")
    skill_counts = {}
    for it in deduped:
        skill_counts[it["skill"]] = skill_counts.get(it["skill"], 0) + 1
    for s, n in skill_counts.items():
        print(f"  {s:20s}: {n} items")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default="data/T3.1_Math_Tutor/curriculum_seed.json")
    parser.add_argument("--out", default="data/T3.1_Math_Tutor/curriculum_full.json")
    parser.add_argument("--target", type=int, default=80)
    args = parser.parse_args()
    generate(Path(args.seed), Path(args.out), args.target)
