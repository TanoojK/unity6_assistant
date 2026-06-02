import shutil
import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig


MODEL_NAME      = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"  
DATASET_FILE    = "./mistral_finetune.jsonl"
ADAPTER_DIR     = "./unity_lora_adapter"
MAX_SEQ_LENGTH  = 4096
LORA_RANK       = 16       # 8 uses less VRAM, 16 gives better quality
LORA_ALPHA      = 32       # usually double of the LoRA_rank is default
LORA_DROPOUT    = 0.05
BATCH_SIZE      = 2        # reduce to 1 if OOM on < 10 GB VRAM
GRAD_ACCUM      = 4        # effective batch = BATCH_SIZE * GRAD_ACCUM = 8
EPOCHS          = 3
LR              = 2e-4
# ─────────────────────────────────────────────────────────────────────────────

def format_chat_sample(sample: dict) -> str:
    """Convert chat messages to Mistral [INST] format."""
    messages = sample["messages"]
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user   = next((m["content"] for m in messages if m["role"] == "user"), "")
    asst   = next((m["content"] for m in messages if m["role"] == "assistant"), "")

    # Mistral instruct template
    if system:
        return f"<s>[INST] {system}\n\n{user} [/INST] {asst}</s>"
    return f"<s>[INST] {user} [/INST] {asst}</s>"


def main():
    print("=" * 60)
    print("Unity 6 Assistant — QLoRA Fine-tuning")
    print("=" * 60)

    # Load base model 
    print("\n[1/5] Loading Mistral 7B Instruct (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,          # auto-detect (bfloat16 on Ampere+, float16 otherwise)
        load_in_4bit=True,
    )

    # Attach LoRA adapters 
    print("\n[2/5] Attaching LoRA adapters (rank={LORA_RANK})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # saves ~30% VRAM
        random_state=42,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"      Trainable params: {trainable/1e6:.1f}M / {total/1e6:.0f}M total ({trainable/total*100:.2f}%)")

    # Load and format dataset 
    print(f"\n[3/5] Loading dataset from {DATASET_FILE}...")
    ds = load_dataset("json", data_files=DATASET_FILE, split="train")
    ds = ds.map(lambda x: {"text": format_chat_sample(x)}, remove_columns=ds.column_names)
    print(f"      {len(ds)} training samples")

    # 4. Train 
    print(f"\n[4/5] Training ({EPOCHS} epochs, lr={LR})...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            save_strategy="epoch",
            output_dir="./training_checkpoints",
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",       # 8-bit Adam saves ~1 GB optimizer memory
            weight_decay=0.01,
            report_to="none",         # disable wandb
        ),
    )
    trainer.train()

    # Save adapter and clean up HF weights 
    print(f"\n[5/5] Saving LoRA adapter to {ADAPTER_DIR}...")
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    adapter_mb = sum(f.stat().st_size for f in os.scandir(ADAPTER_DIR) if f.is_file()) / 1e6
    print(f"      Adapter size: {adapter_mb:.0f} MB")

    # Delete HF cached weights
    hf_cache_paths = [
        os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--mistral-7b-instruct-v0.3-bnb-4bit"),
        os.path.expanduser("~/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3"),
    ]
    for path in hf_cache_paths:
        if os.path.exists(path):
            print(f"      Deleting HF weights: {path}")
            shutil.rmtree(path)
            print(f"      Freed ~4.5 GB")

    # Delete training checkpoints (keep only final adapter)
    if os.path.exists("./training_checkpoints"):
        shutil.rmtree("./training_checkpoints")
        print("      Deleted training checkpoints")

    print("\nDone! Disk usage after cleanup:")
    print(f"  {ADAPTER_DIR}/  →  {adapter_mb:.0f} MB")
    print("\nNext step: run 5_build_rag.py")


if __name__ == "__main__":
    main()
