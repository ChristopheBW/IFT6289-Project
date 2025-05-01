import json
import csv
import requests
import os

OLLAMA_API_BASE = "https://ollama.christophebw.net/api"
OLLAMA_CHAT_URL = f"{OLLAMA_API_BASE}/api/chat"
TEST_DATA_PATH = os.path.join("1_data_preprocessing", "dataset", "empatheticdialogues", "test.jsonl")
OUTPUT_DIR = "3_evaluation/result"

MODELS = ["llama3.2-culture"]

def generate_response_with_history(model, messages):
    """Generate a response from the model using the Ollama chat API with message history."""
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    true_results = []  
    model_results = {model: [] for model in MODELS}  
    
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for idx, line in enumerate(lines):
        record = json.loads(line)
        messages = record["messages"]
        
        if messages[-1]["role"] != "assistant":
            continue
        
        human_response = messages[-1]["content"]
        
        prompt_messages = messages[:-1]
        
        has_system_message = False
        for msg in prompt_messages:
            if msg["role"] == "system":
                msg["content"] = "Keep your answer brief, about 10 to 20 words. " + msg["content"]
                has_system_message = True
                break
            
        
        true_results.append({
            "id": idx,
            "messages": json.dumps(prompt_messages),
            "human_response": human_response
        })
        
        for model in MODELS:
            model_response = generate_response_with_history(model, prompt_messages)
            
            model_results[model].append({
                "id": idx,
                "model_response": model_response
            })
            
            print(f"Processed record {idx} with model {model}")
    
    # Write results to CSV files
    with open(os.path.join(OUTPUT_DIR, "empathy_true_result.csv"), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "messages", "human_response"])
        writer.writeheader()
        writer.writerows(true_results)
    
    for model in MODELS:
        filename = f"empathy_{model}_result.csv"
        with open(os.path.join(OUTPUT_DIR, filename), 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["id", "model_response"])
            writer.writeheader()
            writer.writerows(model_results[model])
    
    print("Complete! CSV files have been created in the output directory.")

if __name__ == "__main__":
    main()
