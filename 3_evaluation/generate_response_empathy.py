import json
import csv
import requests
import os

# API URL
OLLAMA_API_BASE = "https://ollama.christophebw.net/api"
OLLAMA_CHAT_URL = f"{OLLAMA_API_BASE}/chat"
TEST_DATA_PATH = os.path.join("1_data_preprocessing", "dataset", "empatheticdialogues", "test.jsonl")
OUTPUT_DIR = "3_evaluation/result"

# Models to test
MODELS = ["llama3.2", "llama3.2-empathy"]

def generate_response_with_history(model, messages):
    """
    Generate a response from the model using the Ollama chat API with message history.
    """
    data = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    
    response = requests.post(OLLAMA_CHAT_URL, json=data)
    
    if response.status_code == 200:
        return response.json().get("message", {}).get("content", "").strip()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Data to store
    true_results = []  # (id, messages, human_response)
    model_results = {model: [] for model in MODELS}  # (id, model_response)
    
    # Read test data
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for idx, line in enumerate(lines):
        record = json.loads(line)
        messages = record["messages"]
        
        # Skip if the last message is not from assistant
        if messages[-1]["role"] != "assistant":
            continue
        
        # Get the human response (last assistant message)
        human_response = messages[-1]["content"]
        
        # Prepare messages for the model (exclude the last assistant message)
        prompt_messages = messages[:-1]
        
        # Add or update system message with brevity instruction
        has_system_message = False
        for msg in prompt_messages:
            if msg["role"] == "system":
                msg["content"] = "Keep your answer brief, about 10 to 20 words. " + msg["content"]
                has_system_message = True
                break
            
        
        # Store the true result
        true_results.append({
            "id": idx,
            "messages": json.dumps(prompt_messages),  # Store the full message history
            "human_response": human_response
        })
        
        # Generate responses for each model
        for model in MODELS:
            model_response = generate_response_with_history(model, prompt_messages)
            
            model_results[model].append({
                "id": idx,
                "model_response": model_response
            })
            
            print(f"Processed record {idx} with model {model}")
    
    # Write the results to CSV files
    # True results
    with open(os.path.join(OUTPUT_DIR, "empathy_true_result.csv"), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "messages", "human_response"])
        writer.writeheader()
        writer.writerows(true_results)
    
    # Model results
    for model in MODELS:
        filename = f"empathy_{model}_result.csv"
        with open(os.path.join(OUTPUT_DIR, filename), 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["id", "model_response"])
            writer.writeheader()
            writer.writerows(model_results[model])
    
    print("Complete! CSV files have been created in the output directory.")

if __name__ == "__main__":
    main()
