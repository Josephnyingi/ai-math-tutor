"""
train_lora.py — QLoRA fine-tuning of TinyLlama-1.1B-Chat on numeracy feedback pairs.

This script:
  1. Loads data/instruction_data.jsonl (2 000 EN/FR/KIN feedback pairs).
  2. Formats into ChatML prompt template (matches TinyLlama chat format).
  3. Fine-tunes with LoRA (rank=8, alpha=16) using PEFT + bitsandbytes 4-bit quant.
  4. Saves adapter weights to tutor/adapters/tinyllama-numeracy-lora/.
  5. Optionally merges + exports to GGUF via llama.cpp (documented below).

Usage (Colab CPU/GPU or local):
    pip install peft transformers bitsandbytes accelerate datasets trl
    python3 scripts/train_lora.py --epochs 3 --batch 4 --lr 2e-4

For CPU-only (slow but runs):
    python3 scripts/train_lora.py --epochs 1 --batch 1 --no-quant

Estimated time:
    A100 GPU (Colab): ~8 min for 3 epochs on 2 000 samples
    T4 GPU  (Colab): ~25 min
    CPU only       : ~4 h (use --epochs 1 --batch 1 for demo)

After training — merge + quantise to GGUF:
    # 1. Merge LoRA adapter into base weights
    python3 scripts/train_lora.py --merge-only

    # 2. Convert to GGUF with llama.cpp
    git clone https://github.com/ggerganov/llama.cpp
    python3 llama.cpp/convert_hf_to_gguf.py tutor/adapters/merged/ \\
        --outtype q4_k_m --outfile tutor/model.gguf

    # 3. Verify footprint
    du -sh tutor/model.gguf   # target: <75 MB total for tutor/

Model card / weights: https://huggingface.co/Josephnyingi/math-tutor-tinyllama-lora
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = Path("data/instruction_data.jsonl")
OUTPUT_DIR = Path("tutor/adapters/tinyllama-numeracy-lora")
MERGED_DIR = Path("tutor/adapters/merged")


# ------------------------------------------------------------------
# Prompt formatting (TinyLlama ChatML template)
# ------------------------------------------------------------------

def format_prompt(rec: dict) -> str:
    return (
        f"<|system|>\n{rec['instruction']}</s>\n"
        f"<|user|>\n{rec['input']}</s>\n"
        f"<|assistant|>\n{rec['output']}</s>\n"
    )


def load_dataset_records() -> list[dict]:
    records = []
    with open(DATA_PATH, encoding="utf-8") as fh:
        for line in fh:
            records.append(json.loads(line))
    return records


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train(epochs: int = 3, batch_size: int = 4, lr: float = 2e-4, use_quant: bool = True):
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
            BitsAndBytesConfig,
        )
        from trl import SFTTrainer
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install peft transformers bitsandbytes accelerate datasets trl")
        return

    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = None
    if use_quant:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        except Exception:
            print("bitsandbytes 4-bit quant unavailable — falling back to full precision")
            bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    records = load_dataset_records()
    texts = [format_prompt(r) for r in records]
    dataset = Dataset.from_dict({"text": texts})
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 16 // batch_size),
        learning_rate=lr,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        warmup_steps=50,
        lr_scheduler_type="cosine",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=256,
        tokenizer=tokenizer,
        args=training_args,
    )

    print(f"\nTraining for {epochs} epoch(s) on {len(train_ds)} samples...")
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"\nAdapter saved to {OUTPUT_DIR}")
    _print_adapter_size()


def merge(base_model: str = BASE_MODEL):
    """Merge LoRA adapter into base weights and save full model."""
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Run: pip install peft transformers")
        return

    if not OUTPUT_DIR.exists():
        print(f"No adapter found at {OUTPUT_DIR}. Run training first.")
        return

    print("Loading base model for merge...")
    tokenizer = AutoTokenizer.from_pretrained(str(OUTPUT_DIR))
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto")
    model = PeftModel.from_pretrained(base, str(OUTPUT_DIR))
    print("Merging adapter into base weights...")
    merged = model.merge_and_unload()
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(MERGED_DIR))
    tokenizer.save_pretrained(str(MERGED_DIR))
    print(f"Merged model saved to {MERGED_DIR}")
    print("\nNext step — quantise to GGUF:")
    print(f"  python3 llama.cpp/convert_hf_to_gguf.py {MERGED_DIR} --outtype q4_k_m --outfile tutor/model.gguf")


def _print_adapter_size():
    if OUTPUT_DIR.exists():
        size = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
        print(f"Adapter size: {size / 1e6:.1f} MB  ({OUTPUT_DIR})")


def push_to_hub(repo_id: str = "Josephnyingi/math-tutor-tinyllama-lora"):
    """Push adapter weights to Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Run: pip install huggingface_hub peft transformers")
        return

    if not OUTPUT_DIR.exists():
        print(f"No adapter at {OUTPUT_DIR}. Train first.")
        return

    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(OUTPUT_DIR))
    tokenizer.push_to_hub(repo_id)

    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base, str(OUTPUT_DIR))
    model.push_to_hub(repo_id)
    print(f"Adapter pushed to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tune TinyLlama on numeracy feedback")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--no-quant", action="store_true", help="Disable 4-bit quant (for CPU)")
    parser.add_argument("--merge-only", action="store_true", help="Skip training, just merge adapter")
    parser.add_argument("--push", action="store_true", help="Push adapter to Hugging Face Hub")
    args = parser.parse_args()

    if args.merge_only:
        merge()
    elif args.push:
        push_to_hub()
    else:
        train(
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            use_quant=not args.no_quant,
        )
        if args.push:
            push_to_hub()
