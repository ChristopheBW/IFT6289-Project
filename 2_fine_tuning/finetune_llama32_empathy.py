import torch
import os
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

# Model configuration
base_model_id = "meta-llama/Llama-3.2-3B-Instruct"
dataset_dir = "1_data_preprocessing/dataset/empatheticdialogues"
adapter_output_dir = "2_fine_tuning/adapter/llama3.2-empathy-adapters"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

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
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    group_by_length=True,
    report_to="tensorboard",
)

def main():
    """Fine-tunes Llama-3.2-3B model for empathetic conversations."""
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True, use_fast=True, legacy=False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")
    tokenizer.padding_side = "right"

    # Load model
    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    # Load dataset
    print("Loading dataset...")
    train_dataset = load_dataset("json", data_files=os.path.join(dataset_dir, "train.jsonl"), split="train")
    eval_dataset = load_dataset("json", data_files=os.path.join(dataset_dir, "valid.jsonl"), split="train")
    test_dataset = load_dataset("json", data_files=os.path.join(dataset_dir, "test.jsonl"), split="train")
    
    print(f"Dataset loaded with {len(train_dataset)} training samples, {len(eval_dataset)} validation samples, and {len(test_dataset)} test samples.")

    print("First sample structure:")
    print(train_dataset[0])

    # Initialize trainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    # Train model
    print("Starting training...")
    train_result = trainer.train()
    print("Training finished.")

    # Save results
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print("Saving final adapter model...")
    trainer.save_model(adapter_output_dir)

    # Clean up
    del model
    del trainer
    torch.cuda.empty_cache()

    print("--- Fine-tuning Complete ---")
    print(f"Adapters saved to: {adapter_output_dir}")
    print("To use the model, load the base model and apply these adapters.")

if __name__ == "__main__":
    main()

