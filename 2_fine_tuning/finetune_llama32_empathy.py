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

# --- Configuration (Based on finetuning_plan_llama32_chn) ---

# 1. Model Identifier
base_model_id = "meta-llama/Llama-3.2-3B-Instruct" # Confirmed model ID

# 2. Dataset Path
dataset_dir = "1_data_preprocessing/dataset/empatheticdialogues" # Path to empathetic dialogues data

# 3. Output Directory for Adapters
adapter_output_dir = "2_fine_tuning/adapter/llama3.2-empathy-adapters"

# 4. BitsAndBytesConfig (Quantization for QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # As per plan
    bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 if supported, else float16
    bnb_4bit_use_double_quant=True,     # As per plan
)

# 5. LoRA Configuration (PeftConfig)
peft_config = LoraConfig(
    r=16,                               # As per plan (starting point)
    lora_alpha=32,                      # As per plan (starting point, 2*r)
    lora_dropout=0.05,                  # As per plan (starting point)
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[                    # Common targets for Llama models
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        # "gate_proj", # Optional: Add if VRAM allows and needed
        # "up_proj",   # Optional: Add if VRAM allows and needed
        # "down_proj", # Optional: Add if VRAM allows and needed
    ],
)

# 6. Training Arguments
# Effective batch size = per_device_train_batch_size * gradient_accumulation_steps
# Aim for effective batch size of 16 or 32 based on plan
training_args = TrainingArguments(
    output_dir=adapter_output_dir,
    num_train_epochs=3,                 # As per plan
    per_device_train_batch_size=2,      # Adjust based on VRAM (1 or 2 likely)
    gradient_accumulation_steps=8,      # Adjust so batch_size * grad_accum = 16 (or 32)
    optim="paged_adamw_8bit",           # As per plan
    learning_rate=2e-4,                 # As per plan
    lr_scheduler_type="cosine",         # As per plan
    warmup_ratio=0.03,                  # As per plan
    weight_decay=0.01,                  # As per plan
    logging_steps=10,                   # Log training loss frequently
    save_strategy="epoch",              # Save adapter checkpoints every epoch
    #evaluation_strategy="epoch",        # Evaluate on validation set every epoch
    fp16=False,                         # Set fp16=True if bf16=False and CUDA supports fp16 well
    bf16=True,                          # Use bf16 if supported (Ampere GPUs like 3090 support it)
    max_grad_norm=0.3,                  # Helps prevent exploding gradients
    group_by_length=True,               # Group sequences of similar length for efficiency
    report_to="tensorboard",            # Or "wandb" if you prefer
    # load_best_model_at_end=True,      # Optional: Reload best checkpoint based on eval loss
    # metric_for_best_model="loss",     # Optional: Metric to determine best model
)

# --- Script Execution ---

def main():
    # 1. Load Tokenizer and Model
    print("Loading tokenizer...")
    # Ensure you have access or are logged in if model is gated
    # Use legacy=False for Llama 3's tokenizer behavior
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True, use_fast=True, legacy=False)
    # Set padding token if not set (common practice)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")
    # Set padding side to right for training causal LMs
    tokenizer.padding_side = "right"

    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto", # Automatically distribute model layers across available GPUs (or use CPU RAM if needed)
        trust_remote_code=True,
    )
    # Resize token embeddings if pad token was added
    model.resize_token_embeddings(len(tokenizer))

    # Apply PEFT configuration AFTER loading the base model
    # model = get_peft_model(model, peft_config) # SFTTrainer handles this automatically
    # print("PEFT model created:")
    # model.print_trainable_parameters()

    # 2. Load and Prepare Dataset
    print("Loading dataset...")
    train_dataset = load_dataset("json", data_files=os.path.join(dataset_dir, "train.jsonl"), split="train")
    eval_dataset = load_dataset("json", data_files=os.path.join(dataset_dir, "valid.jsonl"), split="train")
    test_dataset = load_dataset("json", data_files=os.path.join(dataset_dir, "test.jsonl"), split="train")
    
    print(f"Dataset loaded with {len(train_dataset)} training samples, {len(eval_dataset)} validation samples, and {len(test_dataset)} test samples.")

    # Optional: Inspect first sample structure
    print("First sample structure:")
    print(train_dataset[0])

    # 3. Initialize SFTTrainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        # ...existing code...
    )

    # 4. Start Training
    print("Starting training...")
    train_result = trainer.train()
    print("Training finished.")

    # 5. Save Metrics and Final Adapter Model
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print("Saving final adapter model...")
    trainer.save_model(adapter_output_dir) # Saves adapters to output_dir specified in TrainingArguments
    # Also saves tokenizer & config files

    # Optional: Clean up GPU memory
    del model
    del trainer
    torch.cuda.empty_cache()

    print("--- Fine-tuning Complete ---")
    print(f"Adapters saved to: {adapter_output_dir}")
    print("To use the model, load the base model and apply these adapters.")

if __name__ == "__main__":
    # Set logging level (optional)
    # logging.set_verbosity_info()
    main()

