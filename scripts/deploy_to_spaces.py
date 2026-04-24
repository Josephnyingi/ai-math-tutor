"""
Deploy the AI Math Tutor to HuggingFace Spaces.

Usage:
    python3 scripts/deploy_to_spaces.py --token hf_XXXX

The script creates (or updates) the Space at:
    https://huggingface.co/spaces/Nyingi101/math-tutor
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

REPO_ID = "Nyingi101/math-tutor"
REPO_TYPE = "space"
SPACE_SDK = "gradio"

# Files/dirs to include in the Space (relative to project root)
INCLUDE = [
    "app.py",
    "demo.py",
    "parent_report.py",
    "tutor/",
    "data/T3.1_Math_Tutor/curriculum_full.json",
    "data/T3.1_Math_Tutor/curriculum_seed.json",
]


def build_staging(project_root: Path, staging: Path) -> None:
    """Copy required files into a clean staging directory."""
    for item in INCLUDE:
        src = project_root / item
        dst = staging / item
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # Space README (replaces root README for Spaces context)
    shutil.copy2(project_root / "README_SPACE.md", staging / "README.md")

    # Spaces requirements (minimal, no eval/notebook deps)
    shutil.copy2(project_root / "requirements_spaces.txt", staging / "requirements.txt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="HuggingFace write token")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    api = HfApi(token=args.token)

    print(f"Creating / verifying Space: {REPO_ID}")
    create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        space_sdk=SPACE_SDK,
        exist_ok=True,
        token=args.token,
    )

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        print("Staging files …")
        build_staging(project_root, staging)

        print(f"Uploading to {REPO_ID} …")
        upload_folder(
            folder_path=str(staging),
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            token=args.token,
            commit_message="Deploy AI Math Tutor",
        )

    print()
    print("Done!")
    print(f"Space URL : https://huggingface.co/spaces/{REPO_ID}")
    print(f"Live demo : https://nyingi101-math-tutor.hf.space")


if __name__ == "__main__":
    main()
