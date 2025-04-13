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
base_model_id = "google/gemma-3-4b-it"  # Changed from Llama-3.2-3B-Instruct to Gemma 3 4B

# 2. Dataset Path
dataset_path = "1_data_preprocessing/dataset/culture_wvs/generated_dialogues/generated_dialogues_batch_CHN.json" # Your generated JSONL file

# 3. Output Directory for Adapters
adapter_output_dir = "./adapter/gemma3-4b-chn-adapters"  # Updated output directory name

# 4. BitsAndBytesConfig (Quantization for QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # As per plan
    bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 if supported, else float16
    #bnb_4bit_use_double_quant=True,     # As per plan
)

# 5. LoRA Configuration (PeftConfig)
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# 6. Training Arguments
# Note: Since Gemma-3-4b is slightly larger than Llama-3.2-3B, batch size or gradient accum may need adjustment
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    max_steps=10,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    output_dir=adapter_output_dir,
    optim="paged_adamw_8bit"
)

# --- Script Execution ---

def main():
    # 1. Load Tokenizer and Model
    print("Loading tokenizer...")
    # Removed legacy=False parameter as it's specific to Llama tokenizers
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True, use_fast=True)
    # Set padding token if not set (common practice)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")
    # Set padding side to right for training causal LMs
    tokenizer.padding_side = "right"

    print(f"Loading {base_model_id} with 4-bit quantization...")
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
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"Dataset loaded with {len(dataset)} samples.")

    # Optional: Inspect first sample structure
    print("First sample structure:")
    print(dataset[0])

    # Split dataset
    print("Splitting dataset...")
    dataset_dict = dataset.train_test_split(test_size=0.05) # 5% validation
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(eval_dataset)}")

    # 3. Initialize SFTTrainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,                        # Base model (PEFT applied internally by trainer)
        #tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,            # Pass PEFT config here
        # dataset_text_field=None,            # Set to None if data is formatted like ChatML (list of dicts)
                                            # Or specify column name if it's just text: "text"
                                            # Or provide a formatting_func
        # packing=False,                      # Set packing=True if formatting data into single sequences
                                            # Set packing=False if using chat format directly
        # max_seq_length=1024,                # Adjust based on VRAM and typical dialogue length (e.g., 512, 1024, 2048)
        # dataset_kwargs={                    # Handles the 'messages' format directly
        #     "add_special_tokens": False,    # We handle special tokens via chat template usually
        #     "append_concat_token": False,   # No need for this with chat format
        # }
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

