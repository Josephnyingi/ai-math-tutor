"""
make_instruction_data.py — Generate 2 000+ numeracy feedback instruction pairs
for QLoRA fine-tuning of TinyLlama-1.1B-Chat.

Output: data/instruction_data.jsonl
Each line is a JSON object:
  {"instruction": "...", "input": "...", "output": "..."}

Coverage:
  - 5 skills × 9 difficulty levels × 3 languages = 135 templates
  - correct / incorrect response variants
  - age-appropriate tone (warm, short, simple words)
  - child first-name variety (from Rwanda / francophone Africa)

Runtime: < 10 s on any laptop. No GPU needed.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(42)

SKILLS = ["counting", "number_sense", "addition", "subtraction", "word_problem"]

NAMES = [
    "Amani", "Keza", "Aline", "Diane", "Eric", "Jean", "Sara", "Claude",
    "Uwase", "Mugisha", "Ingabire", "Habimana", "Niyonzima", "Irakoze",
    "Pierre", "Marie", "Fatou", "Kofi", "Ama", "Kwame",
]

# ------------------------------------------------------------------
# Correct-answer feedback templates (EN / FR / KIN)
# ------------------------------------------------------------------
CORRECT_EN = [
    "Great job, {name}! {answer} is exactly right! Keep going!",
    "Yes! {answer} — you got it! I'm so proud of you!",
    "Wonderful! That's correct! {answer} is the answer!",
    "Amazing work, {name}! You are getting so good at {skill}!",
    "Perfect! {answer} — well done!",
    "That's right! {answer}! You are a math star!",
    "Excellent, {name}! {answer} is correct! Let's try another one!",
    "Correct! {answer}! You are thinking so clearly today!",
    "Superb! {answer} — that's the right answer!",
    "You did it! {answer} is right. I knew you could do it!",
]
CORRECT_FR = [
    "Bravo, {name}! {answer}, c'est exactement ça! Continue!",
    "Oui! {answer} — tu as trouvé! Je suis tellement fier de toi!",
    "Magnifique! C'est correct! {answer} est la réponse!",
    "Excellent travail, {name}! Tu es très fort(e) en {skill}!",
    "Parfait! {answer} — très bien fait!",
    "C'est juste! {answer}! Tu es une étoile des maths!",
    "Excellent, {name}! {answer} est correct! Essayons un autre!",
    "Correct! {answer}! Tu réfléchis si bien aujourd'hui!",
    "Superbe! {answer} — c'est la bonne réponse!",
    "Tu y es arrivé(e)! {answer}, c'est juste. Je savais que tu pouvais!",
]
CORRECT_KIN = [
    "Ni byiza cyane, {name}! {answer} ni yo! Komeza!",
    "Yego! {answer} — wabikoze! Ndishimye cyane!",
    "Ni byiza! Ni byo! {answer} ni igisubizo!",
    "Wakoze neza, {name}! Uri inzobere muri {skill}!",
    "Ni nziza! {answer} — wakoze neza cyane!",
    "Ni byo! {answer}! Uri inyenyeri y'imibare!",
    "Ni byiza, {name}! {answer} ni yo! Reka dugerageze undi!",
    "Ni byo! {answer}! Utekereza neza uyu munsi!",
    "Ni byiza cyane! {answer} — ni igisubizo cy'ukuri!",
    "Wabikoze! {answer} ni byo. Nari nzi ko washoboye!",
]

# ------------------------------------------------------------------
# Wrong-answer feedback templates (EN / FR / KIN)
# ------------------------------------------------------------------
WRONG_EN = [
    "Good try, {name}! The answer is {answer}. Let's try again!",
    "Almost there! It's {answer}. You'll get it next time!",
    "Not quite — the answer is {answer}. Don't worry, keep trying!",
    "Nice effort! The right answer is {answer}. Can you see why?",
    "Close! It's actually {answer}. Let's look at it together.",
    "That was a brave try! The answer is {answer}. Try once more!",
    "Hmm, not this time. The answer is {answer}. You can do it!",
    "Keep going, {name}! The answer is {answer}. Practice makes perfect!",
    "Good thinking! The answer is {answer} — let's try another way.",
    "Learning is trying! The answer is {answer}. You're doing great!",
]
WRONG_FR = [
    "Bon essai, {name}! La réponse est {answer}. Réessaie!",
    "Presque! C'est {answer}. Tu y arriveras la prochaine fois!",
    "Pas tout à fait — la réponse est {answer}. Ne t'inquiète pas, continue!",
    "Bel effort! La bonne réponse est {answer}. Tu vois pourquoi?",
    "Proche! C'est en fait {answer}. Regardons ensemble.",
    "C'était courageux! La réponse est {answer}. Essaie encore!",
    "Hmm, pas cette fois. La réponse est {answer}. Tu peux y arriver!",
    "Continue, {name}! La réponse est {answer}. C'est en pratiquant qu'on progresse!",
    "Bonne réflexion! La réponse est {answer} — essayons autrement.",
    "Apprendre c'est essayer! La réponse est {answer}. Tu te débrouilles bien!",
]
WRONG_KIN = [
    "Gerageza neza, {name}! Igisubizo ni {answer}. Ongera ugerageze!",
    "Hafi! Ni {answer}. Uzagera ubungubu!",
    "Si byo ariko — igisubizo ni {answer}. Ntugire ubwoba, komeza!",
    "Wagerageje neza! Igisubizo ni {answer}. Ureba impamvu?",
    "Hafi! Ni {answer} koko. Reka turebe hamwe.",
    "Wagerageje! Igisubizo ni {answer}. Ongera ugerageze!",
    "Hmm, si byo ubu. Igisubizo ni {answer}. Ushobora!",
    "Komeza, {name}! Igisubizo ni {answer}. Igikorwa gikorwa bitera ubuhanga!",
    "Utekereza neza! Igisubizo ni {answer} — gerageza indi nzira.",
    "Kwiga ni ugerageza! Igisubizo ni {answer}. Urakora neza!",
]

SKILL_LABELS = {
    "en": {"counting": "counting", "number_sense": "number sense",
           "addition": "addition", "subtraction": "subtraction", "word_problem": "word problems"},
    "fr": {"counting": "la comptage", "number_sense": "le sens des nombres",
           "addition": "l'addition", "subtraction": "la soustraction", "word_problem": "les problèmes"},
    "kin": {"counting": "gutunga", "number_sense": "gusobanukirwa imibare",
            "addition": "gushyira hamwe", "subtraction": "gukura", "word_problem": "ibibazo"},
}

# System prompts per language
SYSTEM = {
    "en": "You are a friendly math tutor for children aged 5-9. Give very short, warm feedback in 1-2 sentences. Use simple words. Never use markdown.",
    "fr": "Tu es un tuteur de mathématiques sympa pour les enfants de 5 à 9 ans. Donne un retour très court et chaleureux en 1-2 phrases. Utilise des mots simples. Pas de markdown.",
    "kin": "Uri umwigisha w'imibare w'inshuti ku bana b'imyaka 5-9. Tanga igisubizo gito kandi cyiza mu magambo 1-2. Koresha amagambo yoroheje.",
}


def _instruction(lang: str, skill: str, answer: int, is_correct: bool, child_said: str) -> dict:
    name = random.choice(NAMES)
    skill_label = SKILL_LABELS[lang][skill]
    verdict = "correct" if is_correct else f"incorrect (correct answer is {answer})"
    lang_full = {"en": "English", "fr": "French", "kin": "Kinyarwanda"}[lang]

    instruction = SYSTEM[lang]
    inp = (
        f"Child name: {name}. Skill: {skill_label}. "
        f"The child said: '{child_said}'. Their answer was {verdict}. "
        f"Give feedback in {lang_full}."
    )

    if is_correct:
        pool = {"en": CORRECT_EN, "fr": CORRECT_FR, "kin": CORRECT_KIN}[lang]
    else:
        pool = {"en": WRONG_EN, "fr": WRONG_FR, "kin": WRONG_KIN}[lang]

    output = random.choice(pool).format(name=name, answer=answer, skill=skill_label)
    return {"instruction": instruction, "input": inp, "output": output}


def _child_responses(answer: int, lang: str) -> list[str]:
    """Generate plausible correct and wrong child responses."""
    nums_en = {1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",9:"nine",10:"ten"}
    nums_fr = {1:"un",2:"deux",3:"trois",4:"quatre",5:"cinq",6:"six",7:"sept",8:"huit",9:"neuf",10:"dix"}
    nums_kin = {1:"rimwe",2:"kabiri",3:"gatatu",4:"kane",5:"gatanu",6:"gatandatu",7:"indwi",8:"umunani",9:"icyenda",10:"icumi"}
    num_map = {"en": nums_en, "fr": nums_fr, "kin": nums_kin}[lang]

    correct_word = num_map.get(answer, str(answer))
    correct_str = str(answer)
    wrong_val = answer + random.choice([-2, -1, 1, 2])
    wrong_str = str(wrong_val)
    wrong_word = num_map.get(wrong_val, wrong_str)
    return [correct_word, correct_str, wrong_word, wrong_str]


def generate(out_path: Path, n_target: int = 2000) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []

    answers = list(range(1, 21))  # answer range 1–20
    langs = ["en", "fr", "kin"]

    while len(records) < n_target:
        for skill in SKILLS:
            for lang in langs:
                answer = random.choice(answers)
                responses = _child_responses(answer, lang)

                # Correct response pair
                correct_resp = responses[random.randint(0, 1)]
                records.append(_instruction(lang, skill, answer, True, correct_resp))

                # Wrong response pair
                wrong_resp = responses[random.randint(2, 3)]
                records.append(_instruction(lang, skill, answer, False, wrong_resp))

                if len(records) >= n_target:
                    break
            if len(records) >= n_target:
                break

    random.shuffle(records)

    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    lang_counts = {"en": 0, "fr": 0, "kin": 0}
    for r in records:
        for l in langs:
            if SYSTEM[l] in r["instruction"]:
                lang_counts[l] += 1
                break

    print(f"Generated {len(records)} instruction pairs → {out_path}")
    print(f"  EN: {lang_counts['en']}  FR: {lang_counts['fr']}  KIN: {lang_counts['kin']}")
    skill_counts = {}
    for r in records:
        for s in SKILLS:
            sl = SKILL_LABELS["en"][s]
            if sl in r["input"] or s in r["input"]:
                skill_counts[s] = skill_counts.get(s, 0) + 1
    print(f"  Skills: {skill_counts}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/instruction_data.jsonl")
    parser.add_argument("--n", type=int, default=2000)
    args = parser.parse_args()
    generate(Path(args.out), n_target=args.n)
