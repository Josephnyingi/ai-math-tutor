"""
train_modal.py — Run TinyLlama QLoRA fine-tuning on a free Modal GPU
                 directly from your VSCode terminal.

SETUP (one-time, ~2 minutes):
    pip3 install modal
    modal token new          # opens browser, sign in with GitHub/Google (free)

RUN FROM VSCODE TERMINAL:
    modal run scripts/train_modal.py

    # With custom args:
    modal run scripts/train_modal.py --epochs 3 --hf-token YOUR_HF_TOKEN

WHAT HAPPENS:
    1. Modal provisions a T4 GPU container in the cloud (~30 s)
    2. Installs deps, uploads your instruction dataset
    3. Trains TinyLlama-1.1B with QLoRA (rank=8, 3 epochs, ~25 min on T4)
    4. Saves adapter to /results/tinyllama-numeracy-lora/
    5. Downloads adapter to tutor/adapters/tinyllama-numeracy-lora/  (local)
    6. Optionally pushes to Hugging Face Hub

COST: Modal free tier gives $30/month credit. T4 GPU = $0.000583/s.
      3-epoch training ≈ 25 min = $0.88. Easily within free tier.

AFTER TRAINING:
    # Merge adapter + quantise to GGUF (run locally, CPU is fine):
    python3 scripts/train_lora.py --merge-only
    # Download llama.cpp and convert:
    git clone https://github.com/ggerganov/llama.cpp --depth 1
    pip3 install -r llama.cpp/requirements.txt
    python3 llama.cpp/convert_hf_to_gguf.py tutor/adapters/merged/ \\
        --outtype q4_k_m --outfile tutor/model.gguf
    du -sh tutor/model.gguf   # target: < 75 MB
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

# ------------------------------------------------------------------
# Modal app definition
# ------------------------------------------------------------------

app = modal.App("math-tutor-lora-training")

# GPU image: Python 3.10 + all training deps pre-installed
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.0",
        "transformers==4.40.0",
        "peft==0.10.0",
        "bitsandbytes==0.43.0",
        "accelerate==0.29.0",
        "datasets==2.19.0",
        "trl==0.8.6",
        "scipy",
        "huggingface_hub",
    )
)

# Volume to persist adapter weights between runs
volume = modal.Volume.from_name("math-tutor-adapters", create_if_missing=True)
VOLUME_MOUNT = Path("/results")

# ------------------------------------------------------------------
# Training function — runs on a T4 GPU in the cloud
# ------------------------------------------------------------------

@app.function(
    image=image,
    gpu="T4",
    timeout=3600,        # 1 hour max
    volumes={VOLUME_MOUNT: volume},
    secrets=[
        modal.Secret.from_name("huggingface-token", required=False),
    ],
)
def train_tinyllama(
    instruction_data: bytes,
    epochs: int = 3,
    batch_size: int = 4,
    hf_repo: str = "Josephnyingi/math-tutor-tinyllama-lora",
    push_to_hub: bool = False,
):
    import json, random
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, BitsAndBytesConfig,
    )
    from trl import SFTTrainer

    BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    OUTPUT_DIR = VOLUME_MOUNT / "tinyllama-numeracy-lora"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Decode dataset
    records = [json.loads(l) for l in instruction_data.decode().strip().split("\n")]
    random.seed(42)
    random.shuffle(records)
    print(f"Dataset: {len(records)} instruction pairs")

    def fmt(r):
        return (
            f"<|system|>\n{r['instruction']}</s>\n"
            f"<|user|>\n{r['input']}</s>\n"
            f"<|assistant|>\n{r['output']}</s>\n"
        )

    # Load tokeniser
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4-bit quantisation
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print("Loading TinyLlama-1.1B…")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Dataset
    texts = [fmt(r) for r in records]
    ds = Dataset.from_dict({"text": texts}).train_test_split(test_size=0.05, seed=42)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 16 // batch_size),
        learning_rate=2e-4,
        fp16=True,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        warmup_steps=50,
        lr_scheduler_type="cosine",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        dataset_text_field="text",
        max_seq_length=256,
        tokenizer=tokenizer,
        args=args,
    )

    print(f"\nTraining for {epochs} epoch(s) on {len(ds['train'])} samples…")
    result = trainer.train()
    print(f"\nTraining complete: {result.metrics}")

    # Save final adapter
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    volume.commit()

    adapter_size = sum(
        f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file()
    ) / 1e6
    print(f"\nAdapter saved → {OUTPUT_DIR} ({adapter_size:.1f} MB)")

    # Optional HF push
    hf_token = os.environ.get("HF_TOKEN")
    if push_to_hub and hf_token:
        from huggingface_hub import HfApi
        from peft import PeftModel
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=hf_repo, exist_ok=True)
        model.push_to_hub(hf_repo, token=hf_token)
        tokenizer.push_to_hub(hf_repo, token=hf_token)
        print(f"Pushed to https://huggingface.co/{hf_repo}")
    elif push_to_hub:
        print("HF_TOKEN not set — skipping Hub push. Set it as a Modal secret.")

    # Return adapter bytes for local download
    adapter_path = OUTPUT_DIR / "adapter_model.safetensors"
    return adapter_path.read_bytes() if adapter_path.exists() else b""


# ------------------------------------------------------------------
# Local entrypoint — runs from your VSCode terminal
# ------------------------------------------------------------------

@app.local_entrypoint()
def main(
    epochs: int = 3,
    hf_token: str = "",
    push: bool = False,
):
    data_path = Path("data/instruction_data.jsonl")
    if not data_path.exists():
        print("Dataset not found. Run: python3 scripts/make_instruction_data.py")
        sys.exit(1)

    print(f"Uploading {data_path} ({data_path.stat().st_size // 1024} KB) to Modal...")
    instruction_bytes = data_path.read_bytes()

    print(f"Launching T4 GPU training ({epochs} epochs)...")
    adapter_bytes = train_tinyllama.remote(
        instruction_data=instruction_bytes,
        epochs=epochs,
        push_to_hub=push and bool(hf_token),
        hf_repo="Josephnyingi/math-tutor-tinyllama-lora",
    )

    # Save adapter locally
    if adapter_bytes:
        out = Path("tutor/adapters/tinyllama-numeracy-lora")
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter_model.safetensors").write_bytes(adapter_bytes)
        print(f"\nAdapter downloaded → {out}/adapter_model.safetensors")
        print(f"Size: {len(adapter_bytes) / 1e6:.1f} MB")
        print("\nNext: merge + quantise to GGUF")
        print("  python3 scripts/train_lora.py --merge-only")
    else:
        print("Training complete. Adapter stored in Modal volume 'math-tutor-adapters'.")
        print("Retrieve with: modal volume get math-tutor-adapters tinyllama-numeracy-lora/")
