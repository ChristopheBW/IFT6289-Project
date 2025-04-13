import json
import os
import openai
from tqdm import tqdm
import time
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# Download NLTK tokenizer if not already
nltk.download('punkt')

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_test_data(file_path):
    """Load and preprocess test data from a JSONL file with full message history."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                messages = entry.get("messages", [])
                # Find the last user message and its following assistant reply
                for i in range(len(messages) - 1):
                    if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
                        context = messages[:i+1]  # all messages up to the user message
                        reference = messages[i + 1]["content"].strip()
                        data.append({"context": context, "reference": reference})
                        break
            except json.JSONDecodeError:
                continue
    return data

def format_chat_prompt(messages):
    """Convert message history into OpenAI ChatCompletion-compatible input."""
    return [{"role": m["role"], "content": m["content"]} for m in messages]

def generate_response(model, context_messages, max_retries=3):
    """Generate a response from OpenAI model using context messages."""
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=context_messages,
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message['content'].strip().replace("\n", " ")
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error: {e}. Retrying in 2 seconds... ({attempt+1}/{max_retries})")
                time.sleep(2)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                return "Error generating response"

def calculate_bleu_scores(references, candidates):
    """Compute BLEU-1 to BLEU-4 scores."""
    refs = [[word_tokenize(ref.lower())] for ref in references]
    cands = [word_tokenize(cand.lower()) for cand in candidates]
    smooth = SmoothingFunction().method1

    return {
        "BLEU-1": corpus_bleu(refs, cands, weights=(1, 0, 0, 0), smoothing_function=smooth),
        "BLEU-2": corpus_bleu(refs, cands, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth),
        "BLEU-3": corpus_bleu(refs, cands, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth),
        "BLEU-4": corpus_bleu(refs, cands, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    }

def main():
    # Configuration
    model_name = "gpt-3.5-turbo"  # or your fine-tuned OpenAI model
    test_file = "1_data_preprocessing/dataset/empatheticdiagulogues/test.jsonl"
    output_file = "generated_dialogues_openai.jsonl"

    # Load and prepare test data
    print(f"Loading test data from {test_file}...")
    test_data = load_test_data(test_file)
    print(f"Loaded {len(test_data)} valid conversations for evaluation.")

    # Generate responses
    generated_results = []
    references, candidates = [], []

    for item in tqdm(test_data):
        context = format_chat_prompt(item["context"])
        gold = item["reference"]

        response = generate_response(model_name, context)

        # Save for evaluation
        generated_results.append({
            "context": context,
            "gold_response": gold,
            "generated_response": response
        })

        references.append(gold)
        candidates.append(response)

    # Save outputs
    print(f"Saving outputs to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in generated_results:
            f.write(json.dumps(item) + '\n')

    # BLEU Evaluation
    print("\nCalculating BLEU scores...")
    bleu_scores = calculate_bleu_scores(references, candidates)
    for key, val in bleu_scores.items():
        print(f"{key}: {val:.4f}")

if __name__ == "__main__":
    main()
