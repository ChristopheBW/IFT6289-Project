import pandas as pd
import json
import time
import os
from openai import OpenAI # Use OpenAI library for compatibility
from tqdm import tqdm # For progress bar
import random

# --- Configuration ---
API_KEY_FILE = "apikey.txt"
BASE_URL = "https://api.deepseek.com" # IMPORTANT: Replace
MODEL_NAME = "deepseek-chat" # IMPORTANT: Replace if needed
INPUT_CSV = "1_data_preprocessing/dataset/culture_wvs/majority_answers_CHN_50pct_sum_5pct_close.csv"
OUTPUT_JSONL = "generated_dialogues_batch_CHN.jsonl"
# Number of DIFFERENT dialogues to request in EACH API call (per seed topic)
DIALOGUES_PER_API_CALL = 5 # Adjust (e.g., 3, 5, 10) - balance diversity need vs token limits
DELAY_BETWEEN_CALLS = 1 # Optional delay
# Add slight variation to temperature per call?
RANDOMIZE_TEMP = True
BASE_TEMPERATURE = 0.75
RETRY_PATIENCE = 3  # Number of retry attempts before giving up on a seed topic

# --- Revised System Prompt ---
SYSTEM_PROMPT = """You are an AI assistant tasked with generating **multiple, diverse, and realistic** multi-turn dialogues based on a single cultural context prompt.

**Instructions:**
1.  You will be given the Country, Survey Topic, Common Tendency, and the **number of different dialogues** to generate.
2.  For the provided context, generate the specified number of **distinct** dialogues. Each dialogue should explore **different scenarios, participant roles, or conversation flows** while still subtly reflecting the core cultural tendency.
3.  Each generated dialogue must be natural-sounding, have between 4 and 8 turns, and **implicitly** reflect the cultural tendency (do NOT state the survey details directly).
4.  Your **entire response** MUST be a single JSON object.
5.  This JSON object must contain only one key: `"generated_dialogues"`.
6.  The value of `"generated_dialogues"` must be a **list**, where each element in the list is a JSON object representing **one complete dialogue**.
7.  Each dialogue object in the list must follow the format: `{"messages": [{"role": "user" or "assistant", "content": "utterance"}]}`.

**Example Output Format (for a request asking for 2 dialogues):**
{
  "generated_dialogues": [
    {
      "messages": [
        {"role": "user", "content": "Thinking of taking that overseas assignment..."},
        {"role": "assistant", "content": "Wow! Exciting, but far from your parents?"},
        {"role": "user", "content": "Exactly. That's the hard part."},
        {"role": "assistant", "content": "It's common here to weigh family proximity heavily in such choices."}
      ]
    },
    {
      "messages": [
        {"role": "user", "content": "My sister wants to move across the country for college."},
        {"role": "assistant", "content": "Oh, how are your parents taking it? Being near family is usually a big factor."},
        {"role": "user", "content": "They're trying to be supportive, but you know how much they value having everyone close."},
        {"role": "assistant", "content": "Totally understand. It's a different mindset than just pursuing individual goals."}
      ]
    }
  ]
}
"""

# --- Helper Functions ---

def read_api_key(filepath):
    """Reads the API key from a file."""
    try:
        with open(filepath, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: API key file not found at {filepath}")
        exit()
    except Exception as e:
        print(f"Error reading API key file: {e}")
        exit()

def validate_individual_dialogue_structure(dialogue_obj):
    """Validates the structure of a single dialogue object within the list."""
    if not isinstance(dialogue_obj, dict): return False
    if "messages" not in dialogue_obj: return False
    messages = dialogue_obj["messages"]
    if not isinstance(messages, list): return False
    if not messages: return False # Ensure not empty
    for item in messages:
        if not isinstance(item, dict): return False
        if "role" not in item or "content" not in item: return False
        if item["role"] not in ["user", "assistant"]: return False
        if not isinstance(item["content"], str) or not item["content"]: return False # Ensure content is non-empty string
    # Basic turn count check
    if not (4 <= len(messages) <= 12): # Looser check on received data
         # print(f"Warning: Individual dialogue turn count ({len(messages)}) outside target range (4-8).")
         pass # Accept dialogues slightly outside range if structure is ok
    return True

def validate_batch_response(json_string, expected_count):
    """
    Validates if the string is valid JSON, matches the batch structure,
    and contains roughly the expected number of valid individual dialogues.
    Returns the list of valid dialogue objects if successful, None otherwise.
    """
    try:
        data = json.loads(json_string)
        if not isinstance(data, dict):
            print("Validation Error: Response is not a JSON object.")
            return None
        if "generated_dialogues" not in data:
            print("Validation Error: Missing 'generated_dialogues' key.")
            return None
        dialogue_list = data["generated_dialogues"]
        if not isinstance(dialogue_list, list):
            print("Validation Error: 'generated_dialogues' is not a list.")
            return None

        if not dialogue_list:
            print("Validation Warning: 'generated_dialogues' list is empty.")
            return [] # Return empty list if API provides empty list

        valid_dialogues = []
        for i, dialogue_obj in enumerate(dialogue_list):
            if validate_individual_dialogue_structure(dialogue_obj):
                valid_dialogues.append(dialogue_obj)
            else:
                print(f"Validation Warning: Dialogue at index {i} in the batch has invalid structure. Skipping it.")

        # Check if we got a reasonable number of dialogues back
        if len(valid_dialogues) < expected_count * 0.5: # Allow getting fewer back, but not drastically fewer
             print(f"Validation Warning: Received significantly fewer valid dialogues ({len(valid_dialogues)}) than requested ({expected_count}).")
             # Decide if this is acceptable, maybe return None if too few? For now, accept what's valid.

        if not valid_dialogues:
             print("Validation Error: No valid dialogues found in the 'generated_dialogues' list.")
             return None # Treat as failure if NO valid ones found


        return valid_dialogues # Return the list of *validated* dialogue objects

    except json.JSONDecodeError as e:
        print(f"Validation Error: Failed to decode JSON - {e}")
        # print("------ Received Content Start ------")
        # print(json_string)
        # print("------ Received Content End --------")
        return None
    except Exception as e:
        print(f"Unexpected validation error: {e}")
        return None

# --- Main Script ---

if __name__ == "__main__":
    # 1. Setup
    api_key = read_api_key(API_KEY_FILE)
    if not BASE_URL or BASE_URL == "<URL_HERE>":
         print("Error: Please set the BASE_URL variable in the script.")
         exit()

    client = OpenAI(
        base_url=BASE_URL,
        api_key=api_key
    )

    # 2. Read Input Data
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"Successfully read {len(df)} seed topics from {INPUT_CSV}")
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {INPUT_CSV}")
        exit()
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        exit()

    # 3. Generate Dialogues (One API call per seed topic)
    total_seeds = len(df)
    total_dialogues_generated = 0
    api_call_errors = 0
    validation_failures = 0 # Count API calls that returned invalid structure
    successful_retries = 0 # Count number of successful retries

    print(f"Starting batch dialogue generation...")
    print(f" - Dialogues requested per API call: {DIALOGUES_PER_API_CALL}")
    print(f" - Total API calls planned: {total_seeds}")
    print(f" - Retry attempts per seed: {RETRY_PATIENCE}")
    print(f" - Output file: {OUTPUT_JSONL}")

    with open(OUTPUT_JSONL, 'a', encoding='utf-8') as outfile:
        for index, row in tqdm(df.iterrows(), total=total_seeds, desc="Processing Seeds"):
            country = row['CountryText']
            question = row['QuestionText']
            answer = row['AnswerText']

            user_prompt = f"""Generate {DIALOGUES_PER_API_CALL} different dialogues based on the following cultural context. Ensure diversity between the dialogues as instructed.

Country: {country}
Survey Topic: {question}
Common Tendency: {answer}

Output ONLY the JSON object containing the list of dialogues."""

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            success = False
            retry_count = 0
            
            while not success and retry_count <= RETRY_PATIENCE:
                if retry_count > 0:
                    print(f"\nRetrying seed {index} (attempt {retry_count}/{RETRY_PATIENCE})...")
                
                current_temp = BASE_TEMPERATURE
                if RANDOMIZE_TEMP:
                    current_temp = max(0.1, min(1.0, BASE_TEMPERATURE + random.uniform(-0.1, 0.1)))

                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=current_temp,
                        # response_format={ "type": "json_object" }, # Use if API supports strict JSON mode
                        # max_tokens=2048 # Increase if needed, watch out for limits
                    )
                    response_content = completion.choices[0].message.content

                    # Validate the entire batch response
                    validated_dialogue_list = validate_batch_response(response_content, DIALOGUES_PER_API_CALL)

                    if validated_dialogue_list is not None and validated_dialogue_list: # Validation succeeded and list is not empty
                        for dialogue_obj in validated_dialogue_list:
                            # Write each valid dialogue object as a line
                            outfile.write(json.dumps(dialogue_obj, ensure_ascii=False) + '\n')
                        total_dialogues_generated += len(validated_dialogue_list)
                        success = True
                        if retry_count > 0:
                            successful_retries += 1
                    else:
                        # Validation failed (bad JSON or structure) or empty list
                        if validated_dialogue_list is None:
                            print(f"\nError: API call for seed {index} returned invalid batch structure.")
                            validation_failures += 1
                        else:
                            print(f"\nWarning: API call for seed {index} returned valid structure but no dialogues.")
                        retry_count += 1

                except Exception as e:
                    print(f"\nError during API call for seed {index}: {e}")
                    api_call_errors += 1
                    retry_count += 1

                # Optional delay between attempts (even before retries)
                if DELAY_BETWEEN_CALLS > 0:
                    time.sleep(DELAY_BETWEEN_CALLS)
            
            if not success:
                print(f"\nGiving up on seed {index} after {RETRY_PATIENCE} retry attempts.")

    print("\n--- Generation Summary ---")
    print(f"Total API calls attempted: {total_seeds}")
    print(f" - API call errors: {api_call_errors}")
    print(f" - Responses with invalid structure: {validation_failures}")
    print(f" - Successful retries: {successful_retries}")
    print(f"Successfully generated and wrote {total_dialogues_generated} individual dialogues.")
    print(f"Output saved to {OUTPUT_JSONL}")
