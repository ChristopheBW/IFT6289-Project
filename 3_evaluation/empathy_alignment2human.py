import json
import requests
import time
import csv
import os
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK resources
def download_nltk_resources():
    resources = ['punkt', 'punkt_tab']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Failed to download NLTK resource '{resource}': {e}")

# Call this function at the beginning
download_nltk_resources()

OLLAMA_URL = "https://ollama.christophebw.net/api/chat"
TEST_FILE = "1_data_preprocessing/dataset/empatheticdialogues/test.jsonl"
MODELS = ["llama3.2", "llama3.2-empathy"]
MAX_RETRIES = 3

def load_test_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def send_to_ollama(model, messages):
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(OLLAMA_URL, json={
                "model": model,
                "messages": messages,
                "stream": False
            })
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"[{model}] Error: {e}. Retrying...")
                time.sleep(2)
            else:
                print(f"[{model}] Failed after {MAX_RETRIES} attempts.")
                return "ERROR"

def tokenize(text):
    try:
        return word_tokenize(text.lower())
    except LookupError as e:
        print(f"Warning: NLTK tokenization failed: {e}")
        # Fallback to simple space-based tokenization if NLTK fails
        return text.lower().split()

def calculate_bleu(references, candidates):
    refs = [[tokenize(r)] for r in references]
    cands = [tokenize(c) for c in candidates]
    smooth = SmoothingFunction().method1
    return {
        "BLEU-1": corpus_bleu(refs, cands, weights=(1, 0, 0, 0), smoothing_function=smooth),
        "BLEU-2": corpus_bleu(refs, cands, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth),
        "BLEU-3": corpus_bleu(refs, cands, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth),
        "BLEU-4": corpus_bleu(refs, cands, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth),
    }

def initialize_csv_files():
    """
    Initialize CSV files for each model with headers.
    Returns a dictionary of file writers for each model.
    """
    csv_files = {}
    csv_writers = {}
    
    for model in MODELS:
        output_path = f"Evaluation/responses_{model}.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        file = open(output_path, 'w', newline='', encoding='utf-8')
        writer = csv.writer(file)
        writer.writerow(['ID', 'response_text'])
        
        csv_files[model] = file
        csv_writers[model] = writer
        
    return csv_files, csv_writers
    
def save_response(writer, example_id, response):
    """
    Save a single response to the appropriate CSV file.
    """
    writer.writerow([example_id, response])

def main():
    print(f"Loading test data from {TEST_FILE}...")
    raw_data = load_test_data(TEST_FILE)

    results = {model: [] for model in MODELS}
    references = []
    
    # Initialize CSV files
    csv_files, csv_writers = initialize_csv_files()
    
    try:
        print("Generating responses...\n")
        for idx, ex in enumerate(tqdm(raw_data)):
            full_messages = ex["messages"]
            # Find the last assistant response and the last user message before it
            for i in range(len(full_messages) - 1, -1, -1):
                if full_messages[i]["role"] == "assistant":
                    true_response = full_messages[i]["content"].strip()
                    context = full_messages[:i]  # include messages up to the user message before assistant
                    break
            else:
                continue  # skip if no assistant reply found
    
            references.append(true_response)
            example_id = f"example_{idx}"
    
            for model in MODELS:
                gen_response = send_to_ollama(model, context)
                results[model].append(gen_response)
                # Save response immediately
                save_response(csv_writers[model], example_id, gen_response)
    
        print("\nBLEU Evaluation:\n")
        for model in MODELS:
            print(f"Model: {model}")
            bleu = calculate_bleu(references, results[model])
            for k, v in bleu.items():
                print(f"  {k}: {v:.4f}")
            print()
    finally:
        # Close all CSV files
        for file in csv_files.values():
            file.close()
        print("All CSV files have been saved and closed.")

if __name__ == "__main__":
    main()
