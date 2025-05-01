import torch
import os
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
from transformers.trainer_callback import EarlyStoppingCallback

# --- Configuration ---

# Model and dataset configuration
base_model_id = "meta-llama/Llama-3.2-3B-Instruct"
dataset_path = "1_data_preprocessing/dataset/culture_wvs"
adapter_output_dir = "./adapter/llama3.2-culture-adapters"

# Quantization configuration for QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)

# Training arguments
training_args = TrainingArguments(
    output_dir=adapter_output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    group_by_length=True,
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)

# --- Script Execution ---

def main():
    """Run the fine-tuning process for Llama 3.2 on cultural dataset."""
    wandb.init(project="llama-3.2-culture-finetune", name="llama-3.2-culture-lora")
    
    # Load Tokenizer and Model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True, use_fast=True, legacy=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")
    tokenizer.padding_side = "right"

    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    # Load and Prepare Dataset
    print("Loading pre-split datasets...")
    train_dataset = load_dataset("json", data_files="1_data_preprocessing/dataset/culture_wvs/train_dialogues.jsonl", split="train")
    eval_dataset = load_dataset("json", data_files="1_data_preprocessing/dataset/culture_wvs/val_dialogues.jsonl", split="train")
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(eval_dataset)}")

    print("First training sample structure:")
    print(train_dataset[0])

    # Initialize SFTTrainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    
    # Add EarlyStoppingCallback
    trainer.add_callback(
        EarlyStoppingCallback(
            early_stopping_patience=2,
            early_stopping_threshold=0.01,
        )
    )

    # Start Training
    print("Starting training...")
    train_result = trainer.train()
    print("Training finished.")

    # Save Metrics and Final Adapter Model
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print("Saving final adapter model...")
    trainer.save_model(adapter_output_dir)

    del model
    del trainer
    torch.cuda.empty_cache()

    wandb.finish()
    
    print("--- Fine-tuning Complete ---")
    print(f"Adapters saved to: {adapter_output_dir}")
    print("To use the model, load the base model and apply these adapters.")

if __name__ == "__main__":
    main()

