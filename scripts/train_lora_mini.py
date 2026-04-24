"""
train_lora_mini.py — CPU-friendly LoRA proof-of-concept on distilgpt2.

Purpose: produces REAL committed adapter weights that prove the PEFT pipeline
works end-to-end. distilgpt2 (82 MB) trains in ~3 min on CPU, 1 epoch.

For production (TinyLlama on Colab T4 GPU, ~25 min):
    python3 scripts/train_lora.py --epochs 3

Usage:
    python3 scripts/train_lora_mini.py
"""
from __future__ import annotations

import json
from pathlib import Path

DATA_PATH   = Path("data/instruction_data.jsonl")
OUTPUT_DIR  = Path("tutor/adapters/distilgpt2-numeracy-lora")
BASE_MODEL  = "distilgpt2"

def format_prompt(rec: dict) -> str:
    return f"[INST] {rec['input']} [/INST] {rec['output']}"


def main():
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, DataCollatorForLanguageModeling,
        Trainer,
    )

    print(f"Loading {BASE_MODEL}…")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    # distilgpt2 uses Conv1D layers; PEFT target: c_attn (Q+K+V) and c_proj
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Dataset — use 400 samples to keep CPU runtime ~3 min
    records = []
    with open(DATA_PATH, encoding="utf-8") as fh:
        for line in fh:
            records.append(json.loads(line))
    records = records[:400]

    texts = [format_prompt(r) for r in records]

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=128, padding="max_length"
        )

    ds = Dataset.from_dict({"text": texts}).map(tokenize, batched=True, remove_columns=["text"])
    split = ds.train_test_split(test_size=0.1, seed=42)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("\nTraining LoRA adapter (1 epoch, 360 samples, CPU)…")
    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    adapter_mb = sum(
        f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file()
    ) / 1e6
    print(f"\nAdapter saved → {OUTPUT_DIR}  ({adapter_mb:.1f} MB)")
    print("PEFT pipeline verified end-to-end on CPU.")
    print("\nFor TinyLlama production run (Colab T4, ~25 min):")
    print("  python3 scripts/train_lora.py --epochs 3 --batch 4")


if __name__ == "__main__":
    main()
