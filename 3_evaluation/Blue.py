import csv
import json
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
import os

# Download necessary NLTK resources with error handling
def download_nltk_resources():
    resources = ['punkt', 'punkt_tab']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            print(f"Successfully downloaded NLTK resource: {resource}")
        except Exception as e:
            print(f"Warning: Failed to download NLTK resource '{resource}': {e}")

# Call this function at the beginning
download_nltk_resources()

# Paths to files
TEST_FILE = "1_data_preprocessing/dataset/empatheticdialogues/test.jsonl"
RESPONSE_FILES = {
    "llama3.2": "3_evaluation/responses_llama3.2.csv",
    "llama3.2-empathy": "3_evaluation/responses_llama3.2-empathy.csv"
}

def tokenize(text):
    try:
        return word_tokenize(text.lower())
    except LookupError as e:
        print(f"Warning: NLTK tokenization failed: {e}")
        # Fallback to simple space-based tokenization if NLTK fails
        return text.lower().split()

def calculate_bleu(references, candidates):
    refs = [[tokenize(ref)] for ref in references]
    cands = [tokenize(cand) for cand in candidates]
    smooth = SmoothingFunction().method1
    return {
        "BLEU-1": corpus_bleu(refs, cands, weights=(1, 0, 0, 0), smoothing_function=smooth),
        "BLEU-2": corpus_bleu(refs, cands, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth),
        "BLEU-3": corpus_bleu(refs, cands, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth),
        "BLEU-4": corpus_bleu(refs, cands, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    }

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

def main():
    print("Loading ground-truth responses...")
    references = load_references_from_jsonl(TEST_FILE)

    for model_name, csv_path in RESPONSE_FILES.items():
        print(f"\nEvaluating: {model_name}")
        candidates = load_model_responses(csv_path)
        
        # Ensure equal length
        if len(references) != len(candidates):
            print(f"Warning: {model_name} response count does not match reference count ({len(candidates)} vs {len(references)}). Truncating to match.")
            min_len = min(len(references), len(candidates))
            references = references[:min_len]
            candidates = candidates[:min_len]
        
        bleu = calculate_bleu(references, candidates)
        for k, v in bleu.items():
            print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()
