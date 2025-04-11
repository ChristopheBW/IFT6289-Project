import pandas as pd
import json
import time
import os
from openai import OpenAI # Use OpenAI library for compatibility
from tqdm import tqdm # For progress bar

# --- Configuration ---
API_KEY_FILE = "apikey.txt" # File containing your API key
# IMPORTANT: Replace with your actual API endpoint URL
BASE_URL = "https://api.deepseek.com/"
# IMPORTANT: Specify the model you want to use (e.g., "gpt-4o", "gemini-pro", or the one compatible with your endpoint)
MODEL_NAME = "deepseek-chat" # Replace if needed
# Path to your input CSV file (containing cultural seeds)
INPUT_CSV = "1_data_preprocessing/dataset/culture_wvs/majority_answers_CHN_50pct_sum_5pct_close.csv"
# Desired name for the output JSONL file
OUTPUT_JSONL = "generated_dialogues_CHN.jsonl"
# Number of dialogues to generate for EACH row in the CSV
DIALOGUES_PER_SEED = 5 # Adjust as needed (e.g., 50-200)
# Optional delay between API calls (in seconds) to avoid rate limits
DELAY_BETWEEN_CALLS = 1 # Adjust if you hit rate limits

# --- System Prompt (as defined above) ---
SYSTEM_PROMPT = """You are an AI assistant tasked with generating realistic, multi-turn dialogues. These dialogues should subtly reflect a specific cultural tendency observed in the World Values Survey for a given country and topic.

**Instructions:**
1.  You will be given the Country, the Survey Topic (Question Text), and the Common Tendency (Answer Text).
2.  Generate a natural-sounding dialogue between two or more participants (e.g., friends, family, colleagues).
3.  The dialogue should **implicitly touch upon or reflect** the provided cultural tendency. **Do NOT explicitly state the survey question or answer.** Make it feel like a real conversation where underlying values influence the discussion.
4.  The dialogue should have between 4 and 8 turns in total (a turn consists of one user utterance and one assistant utterance, or sequential utterances by different implied speakers assigned to 'user' and 'assistant' roles for structure).
5.  Your **entire response** MUST be a single JSON object.
6.  The JSON object must contain only one key: `"messages"`.
7.  The value of `"messages"` must be a list of dictionaries.
8.  Each dictionary in the list must have two keys: `"role"` (string: either "user" or "assistant") and `"content"` (string: the utterance).
9.  Vary the scenarios and participant roles for different requests.

**Example Output Format:**
{
  "messages": [
    {"role": "user", "content": "I've been thinking about applying for that promotion."},
    {"role": "assistant", "content": "Oh really? That's a big step. Have you considered how it might affect your time with family?"},
    {"role": "user", "content": "Yeah, that's the main thing holding me back. More responsibility means less flexibility."},
    {"role": "assistant", "content": "It's a tough balance. Around here, most people seem to prioritize family time quite highly, even if it means slower career progression."}
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

def validate_dialogue_json(json_string):
    """
    Validates if the string is valid JSON and matches the expected dialogue structure.
    Returns the parsed dictionary if valid, None otherwise.
    """
    try:
        data = json.loads(json_string)
        if not isinstance(data, dict):
            return None
        if "messages" not in data:
            return None
        messages = data["messages"]
        if not isinstance(messages, list):
            return None
        if not messages: # Ensure not empty
             return None
        for item in messages:
            if not isinstance(item, dict):
                return None
            if "role" not in item or "content" not in item:
                return None
            if item["role"] not in ["user", "assistant"]:
                return None
            if not isinstance(item["content"], str):
                 return None
        # Basic turn count check (adjust range if needed)
        if not (4 <= len(messages) <= 12): # Allow slightly wider range than prompt target
             print(f"Warning: Dialogue turn count ({len(messages)}) outside target range (4-8). Accepting anyway.")
             # return None # Uncomment this line to strictly enforce turn count

        return data # Return the parsed data if validation passes
    except json.JSONDecodeError:
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

    # 3. Generate Dialogues
    total_seeds = len(df)
    total_dialogues_to_generate = total_seeds * DIALOGUES_PER_SEED
    generated_count = 0
    error_count = 0

    print(f"Starting dialogue generation...")
    print(f" - Target dialogues per seed: {DIALOGUES_PER_SEED}")
    print(f" - Total target dialogues: {total_dialogues_to_generate}")
    print(f" - Output file: {OUTPUT_JSONL}")

    # Open file in append mode ('a') to allow resuming partially
    with open(OUTPUT_JSONL, 'a', encoding='utf-8') as outfile:
        # Iterate through each seed topic with a progress bar
        for index, row in tqdm(df.iterrows(), total=total_seeds, desc="Processing Seeds"):
            country = row['CountryText']
            question = row['QuestionText']
            answer = row['AnswerText']

            # Generate N dialogues for this seed
            for i in range(DIALOGUES_PER_SEED):
                user_prompt = f"""Generate a dialogue based on the following cultural context:

Country: {country}
Survey Topic: {question}
Common Tendency: {answer}

Remember to follow all instructions and output ONLY the JSON object."""

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]

                try:
                    # Make the API call
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=0.7, # Adjust temperature for creativity vs consistency
                        # max_tokens=500 # Optional: Set max tokens if needed
                    )

                    response_content = completion.choices[0].message.content
                    print("DEBUG: Raw API response content:", response_content) 

                    # Validate the response
                    validated_data = validate_dialogue_json(response_content)

                    if validated_data:
                        # Write the validated JSON object (as a string) to the file, followed by a newline
                        # We dump the validated data to ensure consistent JSON formatting
                        outfile.write(json.dumps(validated_data, ensure_ascii=False) + '\n')
                        generated_count += 1
                    else:
                        print(f"\nWarning: Invalid JSON or structure received for seed {index}, attempt {i+1}. Skipping.")
                        # print("------ Received Content Start ------")
                        # print(response_content)
                        # print("------ Received Content End --------")
                        error_count += 1

                except Exception as e:
                    print(f"\nError during API call or processing for seed {index}, attempt {i+1}: {e}")
                    error_count += 1
                    # Optional: Add more robust error handling (e.g., retry logic)

                # Optional delay
                if DELAY_BETWEEN_CALLS > 0:
                    time.sleep(DELAY_BETWEEN_CALLS)

    print("\n--- Generation Summary ---")
    print(f"Successfully generated and wrote {generated_count} dialogues.")
    print(f"Encountered {error_count} errors or invalid responses.")
    print(f"Output saved to {OUTPUT_JSONL}")

