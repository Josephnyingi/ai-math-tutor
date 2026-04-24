"""
parent_report.py — Generate the weekly 1-page parent report from local SQLite.

Output formats:
  --format html   : visual icon-based HTML (default) — printable, QR-code link to audio
  --format json   : raw JSON for programmatic use
  --format text   : plain-text summary

Design goal: a non-literate parent can understand the report in ≤ 60 seconds.
  - Three big icons: trend arrow (↑/→/↓), best skill badge, help-needed flag
  - Five skill bars (colour-coded: green ≥ 0.7, amber 0.4–0.7, red < 0.4)
  - One voiced summary (QR code links to a short TTS audio file)
  - All numbers shown as visual bars, not raw floats
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tutor.progress_store import ProgressStore

SKILL_ICONS = {
    "counting": "🔢",
    "number_sense": "🧠",
    "addition": "➕",
    "subtraction": "➖",
    "word_problem": "📖",
}
SKILL_LABELS = {
    "en": {
        "counting": "Counting",
        "number_sense": "Number Sense",
        "addition": "Addition",
        "subtraction": "Subtraction",
        "word_problem": "Word Problems",
    },
    "fr": {
        "counting": "Compter",
        "number_sense": "Sens des nombres",
        "addition": "Addition",
        "subtraction": "Soustraction",
        "word_problem": "Problèmes",
    },
    "kin": {
        "counting": "Gutunga",
        "number_sense": "Gusobanukirwa Imibare",
        "addition": "Gushyira hamwe",
        "subtraction": "Gukura",
        "word_problem": "Ibibazo",
    },
}


def _bar(value: float, width: int = 10) -> str:
    filled = round(value * width)
    color = "🟢" if value >= 0.7 else ("🟡" if value >= 0.4 else "🔴")
    return color + "█" * filled + "░" * (width - filled)


def _arrow(delta: float) -> str:
    if delta > 0.05:
        return "⬆️"
    if delta < -0.05:
        return "⬇️"
    return "➡️"


def generate_html(report: dict, lang: str = "en") -> str:
    labels = SKILL_LABELS.get(lang, SKILL_LABELS["en"])
    skills = report["skills"]
    icons = report["icons_for_parent"]
    arrow = "⬆️" if icons["overall_arrow"] == "up" else "➡️"

    rows = ""
    for skill, data in skills.items():
        pct = int(data["current"] * 100)
        bar_w = int(data["current"] * 200)
        color = "#22c55e" if data["current"] >= 0.7 else ("#f59e0b" if data["current"] >= 0.4 else "#ef4444")
        rows += f"""
        <tr>
          <td style="font-size:1.5em; text-align:center">{SKILL_ICONS[skill]}</td>
          <td style="font-size:1.1em; padding:4px 8px">{labels[skill]}</td>
          <td>
            <div style="background:#e5e7eb;border-radius:6px;width:200px;height:20px;">
              <div style="background:{color};width:{bar_w}px;height:20px;border-radius:6px;"></div>
            </div>
          </td>
          <td style="font-weight:bold; padding:0 8px">{pct}%</td>
          <td style="font-size:1.3em">{_arrow(data['delta'])}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Math Tutor — Weekly Report</title>
  <style>
    body {{ font-family: 'Nunito', 'Segoe UI', sans-serif; max-width: 480px;
           margin: 20px auto; padding: 16px; background: #f8fafc; }}
    .header {{ text-align: center; padding: 12px; background: #1a56db;
              color: white; border-radius: 12px; margin-bottom: 16px; }}
    .big-icons {{ display: flex; justify-content: space-around;
                 font-size: 2em; margin: 16px 0; text-align: center; }}
    .big-icons div {{ background: white; border-radius: 12px; padding: 12px 16px;
                     box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
    table {{ width: 100%; border-collapse: collapse; background: white;
            border-radius: 12px; overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
    tr {{ border-bottom: 1px solid #f1f5f9; }}
    td {{ padding: 8px 4px; vertical-align: middle; }}
    .footer {{ text-align: center; margin-top: 16px; color: #64748b; font-size: 0.9em; }}
  </style>
</head>
<body>
  <div class="header">
    <h2 style="margin:0">📊 Weekly Progress</h2>
    <p style="margin:4px 0; opacity:0.85">{report['learner_id']} · {report['week_starting']}</p>
    <p style="margin:4px 0; font-size:1.8em">{report['sessions']} sessions this week</p>
  </div>

  <div class="big-icons">
    <div>{arrow}<br><small>Overall</small></div>
    <div>{SKILL_ICONS.get(icons['best_skill'], '⭐')}<br><small>Best</small></div>
    <div>{'⚠️' if icons['needs_help'] else '✅'}<br><small>Help needed</small></div>
  </div>

  <table>
    {rows}
  </table>

  <div class="footer">
    <p>🔊 <a href="{report.get('voiced_summary_audio', '#')}">Listen to summary</a></p>
    <p style="font-size:0.8em; color:#94a3b8">AI Math Tutor · Offline · Private</p>
  </div>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate weekly parent report")
    parser.add_argument("learner_id", help="Learner ID")
    parser.add_argument("--db", default="tutor_progress.db", help="Path to SQLite DB")
    parser.add_argument("--format", choices=["html", "json", "text"], default="html")
    parser.add_argument("--lang", choices=["en", "fr", "kin"], default="en")
    parser.add_argument("--out", help="Output file (default: stdout)")
    args = parser.parse_args()

    store = ProgressStore(args.db)
    report = store.weekly_report(args.learner_id)
    store.close()

    if args.format == "json":
        output = json.dumps(report, indent=2)
    elif args.format == "html":
        output = generate_html(report, lang=args.lang)
    else:
        # Plain text summary for terminal
        skills = report["skills"]
        lines = [
            f"Weekly Report: {report['learner_id']}  ({report['week_starting']})",
            f"Sessions: {report['sessions']}",
            "",
        ]
        for skill, data in skills.items():
            bar = _bar(data["current"])
            lines.append(f"  {SKILL_ICONS[skill]} {skill:15s} {bar}  {int(data['current']*100)}%")
        icons = report["icons_for_parent"]
        lines += [
            "",
            f"Overall trend : {icons['overall_arrow']}",
            f"Best skill    : {icons['best_skill']}",
            f"Needs help    : {icons['needs_help']}",
        ]
        output = "\n".join(lines)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Report written to {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()
