import json
import os
from pathlib import Path

# Input and output paths
INPUT_DIR = Path("1_data_preprocessing/dataset/culture_wvs/generated_dialogues")
TRAIN_OUTPUT_FILE = Path("1_data_preprocessing/dataset/culture_wvs/train_dialogues.jsonl")
VAL_OUTPUT_FILE = Path("1_data_preprocessing/dataset/culture_wvs/val_dialogues.jsonl")
COUNTRY_CODES = ["CAN", "CHN", "GBR", "IDN"]

# System prompts for each culture
SYSTEM_PROMPTS = {
    "CAN": "You are an AI assistant simulating a perspective reflective of common values and attitudes found in Canadian culture. Respond naturally to the user's situation from this cultural viewpoint, drawing on typical priorities and social norms associated with Canadian culture.",
    "CHN": "You are an AI assistant simulating a perspective reflective of common values and attitudes found in Chinese culture. Respond naturally to the user's situation from this cultural viewpoint, drawing on typical priorities and social norms associated with Chinese culture.",
    "GBR": "You are an AI assistant simulating a perspective reflective of common values and attitudes found in British culture. Respond naturally to the user's situation from this cultural viewpoint, drawing on typical priorities and social norms associated with British culture.",
    "IDN": "You are an AI assistant simulating a perspective reflective of common values and attitudes found in Indonesian culture. Respond naturally to the user's situation from this cultural viewpoint, drawing on typical priorities and social norms associated with Indonesian culture."
}

def main():
    """Combines dialogue data from multiple countries into train and validation sets."""
    train_dialogues = []
    val_dialogues = []

    for country_code in COUNTRY_CODES:
        input_file = INPUT_DIR / f"generated_dialogues_batch_{country_code}.jsonl"

        if not input_file.exists():
            print(f"Warning: {input_file} not found, skipping.")
            continue

        print(f"Processing {country_code} dialogues...")

        with open(input_file, 'r', encoding='utf-8') as f:
            dialogues_in_file = [json.loads(line.strip()) for line in f]

        for i, dialogue_data in enumerate(dialogues_in_file):
             messages = dialogue_data["messages"]
             messages.insert(0, {
                 "role": "system",
                 "content": SYSTEM_PROMPTS[country_code]
             })
             dialogue_data["messages"] = messages
             
             if (i + 1) % 5 == 0:
                 val_dialogues.append(dialogue_data)
             else:
                 train_dialogues.append(dialogue_data)

    TRAIN_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    VAL_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing {len(train_dialogues)} training dialogues to {TRAIN_OUTPUT_FILE}...")
    with open(TRAIN_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for dialogue in train_dialogues:
            f.write(json.dumps(dialogue, ensure_ascii=False) + '\n')

    print(f"Writing {len(val_dialogues)} validation dialogues to {VAL_OUTPUT_FILE}...")
    with open(VAL_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for dialogue in val_dialogues:
            f.write(json.dumps(dialogue, ensure_ascii=False) + '\n')

    print(f"Processed into {len(train_dialogues)} training dialogues and {len(val_dialogues)} validation dialogues")
    print(f"Training data written to {TRAIN_OUTPUT_FILE}")
    print(f"Validation data written to {VAL_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
