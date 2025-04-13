import json
import csv
from bert_score import score
import argparse
import os

# File paths
TEST_FILE = "1_data_preprocessing/dataset/empatheticdialogues/test.jsonl"
RESPONSE_FILES = {
    "llama3.2": "3_evaluation/responses_llama3.2.csv",  # Fixed typo: 3_valuation -> 3_evaluation
    "llama3.2-empathy": "3_evaluation/responses_llama3.2-empathy.csv"
}

# Check and display actual paths
def check_file_paths():
    for model_name, path in RESPONSE_FILES.items():
        if os.path.exists(path):
            print(f"Found response file for {model_name}: {path}")
        else:
            print(f"Warning: Response file not found for {model_name}: {path}")
            # Try alternative path
            alt_path = f"3_evaluation/responses_{model_name}.csv"
            if os.path.exists(alt_path):
                print(f"Found alternative path: {alt_path}")
                RESPONSE_FILES[model_name] = alt_path

def load_references_from_jsonl(file_path):
    references = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            messages = entry["messages"]
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "assistant":
                    references.append(messages[i]["content"].strip())
                    break
    return references

def load_model_responses(csv_file):
    responses = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            responses.append(row["response_text"].strip())
    return responses

def evaluate_bertscore(candidates, references, lang="en", model_type="roberta-large"):
    # Suppress warnings during initialization
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    P, R, F1 = score(candidates, references, lang=lang, model_type=model_type)
    return {
        "Precision": P.mean().item(),
        "Recall": R.mean().item(),
        "F1": F1.mean().item()
    }

def main():
    print("Loading ground-truth responses...")
    check_file_paths()
    
    references = load_references_from_jsonl(TEST_FILE)

    for model_name, csv_path in RESPONSE_FILES.items():
        print(f"\nEvaluating BERTScore for: {model_name}")
        try:
            candidates = load_model_responses(csv_path)

            # Match lengths
            if len(candidates) != len(references):
                min_len = min(len(candidates), len(references))
                candidates = candidates[:min_len]
                references = references[:min_len]

            results = evaluate_bertscore(candidates, references)
            for k, v in results.items():
                print(f"  {k}: {v:.4f}")
        except FileNotFoundError as e:
            print(f"Error: Could not find response file for {model_name} at {csv_path}")
            print(f"Exception: {e}")

if __name__ == "__main__":
    main()
