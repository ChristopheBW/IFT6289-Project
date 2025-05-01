import pandas as pd
import requests
import json
import os
import time
from typing import Dict, List, Any, Tuple

# Ollama API endpoint
OLLAMA_API_URL = "https://ollama.christophebw.net/api/generate"

# Models to use
MODELS = ["llama3.2", "llama3.2-culture"]

# Culture to country code mapping
CULTURES = {
    "Canada": "CAN",
    "Chinese": "CHN",
    "Great Britain": "GBR",
    "Indonesia": "IDN"
}

def create_system_prompt(culture: str) -> str:
    """Create a system prompt for a specific culture."""
    return f"""You are an AI assistant tasked with responding to survey questions from a specific cultural perspective. 
For the following questions, which are part of the Values Survey Module (VSM), adopt the persona of someone 
whose values, attitudes, and beliefs are broadly representative of {culture} culture.

Answer each question based on how you believe a typical individual deeply embedded in {culture} culture might respond. 
Your response should reflect common {culture} priorities, social norms, and underlying cultural values.
Focus on expressing the reasoning or typical viewpoint associated with that culture when answering.

IMPORTANT: Your answer must ONLY contain a single number corresponding to the scale in the question.
Do not include any explanations or additional text in your response."""

def generate_response(model: str, system_prompt: str, question: str, temperature: float = 0.7, seed: int = None) -> Dict[str, Any]:
    """Generate a response using the Ollama API."""
    payload = {
        "model": model,
        "prompt": question,
        "system": system_prompt,
        "stream": False,
        "temperature": temperature
    }
    
    # Add seed if provided
    if seed is not None:
        payload["seed"] = seed
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return {"error": str(e)}

def extract_numeric_answer(response_text: str) -> str:
    """Extract just the numeric answer from the response."""
    # First try to get just the first character if it's a digit
    if response_text and response_text[0].isdigit():
        return response_text[0]
    
    # If that fails, look for any digit in the response
    for char in response_text:
        if (char.isdigit()):
            return char
            
    return "N/A"  # Return N/A if no digit is found

def generate_response_with_retry(model: str, system_prompt: str, question: str, max_retries: int = 3, temperature: float = 0.7, seed: int = None) -> Tuple[Dict[str, Any], str]:
    """Generate a response using the Ollama API with retry logic."""
    for attempt in range(max_retries):
        response_data = generate_response(model, system_prompt, question, temperature=temperature, seed=seed)
        
        if "error" in response_data:
            print(f"    Attempt {attempt+1} failed with error")
            continue
            
        full_response = response_data.get("response", "")
        answer = extract_numeric_answer(full_response)
        
        # Print the response for monitoring
        print(f"    Response: '{full_response}' → Extracted answer: '{answer}'")
        
        if answer != "N/A":
            # Add seed to response data
            response_data["seed"] = seed
            return response_data, answer
            
        print(f"    Attempt {attempt+1} gave non-numeric answer, retrying...")
        time.sleep(1)  # Wait a bit before retrying
    
    # If we get here, all retries failed
    return {"error": "All retries failed", "seed": seed}, "N/A"

def generate_responses_with_seeds(model: str, system_prompt: str, question: str, num_seeds: int = 100, temperature: float = 0.7) -> List[Tuple[Dict[str, Any], str]]:
    """Generate multiple responses using different seeds for diversity.
    
    Note: This uses both seed variation and temperature to increase diversity.
    The temperature setting has a more reliable effect on generating diverse responses.
    """
    results = []
    
    for i in range(num_seeds):
        seed = 1000 + i  # Using larger seed values to avoid potential collisions
        print(f"    Run {i+1}/{num_seeds} with seed {seed} and temperature {temperature}")
        
        response_data, answer = generate_response_with_retry(
            model, 
            system_prompt, 
            question, 
            temperature=temperature, 
            seed=seed
        )
        
        results.append((response_data, answer))
        
        # Add a small delay between runs
        time.sleep(0.5)
    
    return results

def main():
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
    os.makedirs(output_dir, exist_ok=True)
    
    # Read questions from CSV
    questions_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                 "formatted_vsm_questions.csv")
    questions_df = pd.read_csv(questions_file)
    
    # Dictionary to store all results
    all_results = {}
    
    for model in MODELS:
        model_results = {}
        
        for culture_name, country_code in CULTURES.items():
            print(f"Processing {culture_name} culture with {model} model...")
            culture_results = []
            system_prompt = create_system_prompt(culture_name)
            
            for _, row in questions_df.iterrows():
                question_id = row['id']
                question_text = row['question']
                
                print(f"  - Question {question_id}")
                
                # Generate multiple responses with different seeds
                seed_responses = generate_responses_with_seeds(model, system_prompt, question_text)
                
                for response_data, answer in seed_responses:
                    if "error" in response_data:
                        full_response = response_data["error"]
                    else:
                        full_response = response_data.get("response", "")
                    
                    culture_results.append({
                        "country_code": country_code,
                        "question_id": int(question_id),
                        "seed": response_data.get("seed", -1),
                        "answer": answer,
                        "full_response": full_response
                    })
            
            model_results[culture_name] = culture_results
        
        all_results[model] = model_results
    
    # Save results to JSON file
    results_file = os.path.join(output_dir, "vsm_responses.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    # Create one CSV file per model with all culture responses
    for model in MODELS:
        # Group results by country and question to calculate means
        grouped_data = {}
        
        for culture_name, country_code in CULTURES.items():
            for result in all_results[model][culture_name]:
                # Convert answer to integer, if not possible, use -1
                try:
                    answer_int = int(result['answer'])
                except (ValueError, TypeError):
                    answer_int = -1
                
                # Create a key for grouping
                key = (result['country_code'], result['question_id'])
                
                if key not in grouped_data:
                    grouped_data[key] = {
                        "country_code": result['country_code'],
                        "question_id": result['question_id'],
                        "answers": [],
                        "seeds": []
                    }
                
                grouped_data[key]["answers"].append(answer_int)
                grouped_data[key]["seeds"].append(result.get('seed', -1))
        
        # Create CSV data with means
        csv_data = []
        for key, data in grouped_data.items():
            valid_answers = [a for a in data["answers"] if a >= 0]
            
            # Calculate mean only if there are valid answers
            if valid_answers:
                mean_answer = sum(valid_answers) / len(valid_answers)
            else:
                mean_answer = -1
            
            # Format the answers list as a string
            answers_str = ','.join(map(str, data["answers"]))
            seeds_str = ','.join(map(str, data["seeds"]))
            
            csv_data.append({
                "country_code": data["country_code"],
                "question_id": data["question_id"],
                "answer": round(mean_answer, 2),  # Mean rounded to 2 decimal places
                "answers": answers_str,  # All individual answers
                "seeds": seeds_str  # All seeds used
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_file = os.path.join(output_dir, f"{model}_responses.csv")
        csv_df.to_csv(csv_file, index=False)
        print(f"CSV summary for {model} saved to {csv_file}")

if __name__ == "__main__":
    main()
