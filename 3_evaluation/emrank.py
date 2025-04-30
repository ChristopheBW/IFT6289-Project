import pandas as pd
import json
import os
import random
import time
import re
from openai import OpenAI
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuration ---
API_KEY_FILE = "apikey.txt"  # Path to your DeepSeek API key file
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"
SAMPLE_SIZE = 450 # Number of records to evaluate
OUTPUT_DIR = "3_evaluation/"
RESULTS_CSV = os.path.join(OUTPUT_DIR, "emrank_deepseek_results.csv")
PLOT_PNG = os.path.join(OUTPUT_DIR, "emrank_deepseek_summary.png")

# File paths for input data
TRUE_DATA_PATH = "3_evaluation/result/empathy_true_result.csv"
MODEL_A_PATH = "3_evaluation/result/empathy_llama3.2_result.csv" # Model A
MODEL_B_PATH = "3_evaluation/result/empathy_llama3.2-empathy_result.csv" # Model B
MODEL_A_NAME = "llama3.2"
MODEL_B_NAME = "llama3.2-empathy"

# --- API Key Handling ---
def read_api_key(filepath):
    """Reads the API key from a file."""
    try:
        with open(filepath, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: API key file not found at {filepath}")
        print("Please create 'apikey.txt' with your DeepSeek API key.")
        exit()
    except Exception as e:
        print(f"Error reading API key file: {e}")
        exit()

# --- DeepSeek Client Initialization ---
try:
    api_key = read_api_key(API_KEY_FILE)
    client = OpenAI(
        base_url=BASE_URL,
        api_key=api_key
    )
    print("DeepSeek client initialized successfully.")
except Exception as e:
    print(f"Failed to initialize DeepSeek client: {e}")
    exit()

# --- Data Loading and Preparation ---
def load_and_prepare_data(true_path, model_a_path, model_b_path):
    """Loads and merges the datasets."""
    try:
        df_true = pd.read_csv(true_path)
        df_model_a = pd.read_csv(model_a_path)
        df_model_b = pd.read_csv(model_b_path)
        print(f"Loaded {len(df_true)} true records, {len(df_model_a)} model A records, {len(df_model_b)} model B records.")

        # Rename columns for clarity before merging
        df_model_a = df_model_a.rename(columns={'model_response': 'response_a'})
        df_model_b = df_model_b.rename(columns={'model_response': 'response_b'})

        # Merge dataframes
        df_merged = pd.merge(df_true, df_model_a, on='id', how='inner')
        df_merged = pd.merge(df_merged, df_model_b, on='id', how='inner')
        print(f"Merged data contains {len(df_merged)} records.")

        if len(df_merged) == 0:
             print("Error: Merged DataFrame is empty. Check 'id' columns and file contents.")
             exit()

        return df_merged

    except FileNotFoundError as e:
        print(f"Error loading data file: {e}")
        exit()
    except Exception as e:
        print(f"Error processing data: {e}")
        exit()

# --- Context Formatting ---
def format_conversation_context(messages_str):
    """Parses the JSON string in 'messages' and formats it for the prompt."""
    try:
        messages = json.loads(messages_str)
        formatted_context = ""
        for msg in messages:
            role = msg.get('role', 'unknown').capitalize()
            content = msg.get('content', '').replace('_comma_', ',') # Replace placeholder
            formatted_context += f"{role}: {content}\n"
        # Return the last user message as the primary 'Patient Question' for the prompt structure
        # and the full history as context. Or adjust as needed.
        last_user_message = "User query not found."
        for msg in reversed(messages):
             if msg.get('role') == 'user':
                 last_user_message = msg.get('content', '').replace('_comma_', ',')
                 break
        # For this task, providing the full history might be better
        # return formatted_context.strip()
        return formatted_context.strip() # Use full history
    except json.JSONDecodeError:
        print(f"Warning: Could not parse messages JSON: {messages_str}")
        return "Error parsing context."
    except Exception as e:
        print(f"Warning: Error formatting context: {e}")
        return "Error formatting context."


# --- EMRank Evaluation Function ---
def evaluate_empathy(context, response_a, response_b, model_a_name, model_b_name):
    """Uses DeepSeek API to evaluate which response is more empathetic."""
    system_prompt = (
        "You are an expert in comparing the empathy level of two responses to a patient question or conversation history. "
        "You will be given a conversation history ending with a user message, and two possible responses. "
        "Your task is to evaluate which response is more empathetic in the context of the conversation. "
        "Consider factors like understanding, validation of feelings, offering support, and appropriateness."
    )
    user_prompt = (
        f"Conversation History:\n------\n{context}\n------\n\n"
        f"Response 1 ({model_a_name}):\n{response_a}\n\n"
        f"Response 2 ({model_b_name}):\n{response_b}\n\n"
        "Which response is more empathetic? Respond with 'Response 1' or 'Response 2' followed by your reasoning. "
        "Start your answer directly with 'Response 1' or 'Response 2'."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    print("\n--- Sending Request to DeepSeek ---")
    print(f"Context sample: {context[:200]}...") # Print sample context
    print(f"Response 1 ({model_a_name}): {response_a}")
    print(f"Response 2 ({model_b_name}): {response_b}")
    # print(f"Full User Prompt:\n{user_prompt}") # Uncomment for full prompt debugging

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=500, # Adjust as needed
            temperature=0.1 # Low temperature for consistent evaluation
        )
        result_text = response.choices[0].message.content.strip()
        print("\n--- Received Response from DeepSeek ---")
        print(result_text)

        # Parse the result
        judgement = "Unknown"
        reasoning = result_text
        # Simple parsing: check the beginning of the response
        if result_text.lower().startswith("response 1"):
            judgement = "Response 1"
        elif result_text.lower().startswith("response 2"):
            judgement = "Response 2"
        else:
            # Try regex as a fallback (more robust)
            match = re.search(r"^\s*Response\s*([12])", result_text, re.IGNORECASE)
            if match:
                judgement = f"Response {match.group(1)}"
            else:
                 print("Warning: Could not parse judgement from response.")


        return judgement, reasoning

    except Exception as e:
        print(f"Error calling DeepSeek API: {e}")
        return "API Error", str(e)

# --- Main Execution ---
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    df_merged = load_and_prepare_data(TRUE_DATA_PATH, MODEL_A_PATH, MODEL_B_PATH)

    # Sample data
    if len(df_merged) < SAMPLE_SIZE:
        print(f"Warning: Available data ({len(df_merged)}) is less than sample size ({SAMPLE_SIZE}). Using all available data.")
        df_sampled = df_merged
    else:
        df_sampled = df_merged.sample(n=SAMPLE_SIZE, random_state=42) # Use random_state for reproducibility
        print(f"Sampled {len(df_sampled)} records for evaluation.")

    # Evaluate samples
    results = []
    print(f"\nStarting EMRank evaluation for {len(df_sampled)} samples...")
    # Use tqdm for progress bar
    for index, row in tqdm(df_sampled.iterrows(), total=len(df_sampled), desc="Evaluating Empathy"):
        context = format_conversation_context(row['messages'])
        response_a = row['response_a']
        response_b = row['response_b']
        human_response = row['human_response'] # Keep human response for reference if needed

        if context == "Error parsing context." or context == "Error formatting context.":
            print(f"Skipping row {row['id']} due to context formatting error.")
            continue

        judgement, reasoning = evaluate_empathy(context, response_a, response_b, MODEL_A_NAME, MODEL_B_NAME)

        results.append({
            'id': row['id'],
            'context': context,
            'response_a': response_a,
            'response_b': response_b,
            'llm_judgement': judgement,
            'llm_reasoning': reasoning
        })

        # Optional: Add a small delay to avoid hitting rate limits
        time.sleep(1) # Adjust delay as needed (e.g., 0.5 or 1 second)

    print("\nEvaluation finished.")

    # Save results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_CSV, index=False)
    print(f"Results saved to {RESULTS_CSV}")

    # --- Plotting ---
    if not df_results.empty and 'llm_judgement' in df_results.columns:
        # Count judgements, excluding errors/unknowns
        judgement_counts = df_results['llm_judgement'].value_counts()
        count_a = judgement_counts.get("Response 1", 0)
        count_b = judgement_counts.get("Response 2", 0)
        count_total_valid = count_a + count_b

        if count_total_valid > 0:
            percent_a = (count_a / count_total_valid) * 100
            percent_b = (count_b / count_total_valid) * 100

            labels = [f'{MODEL_A_NAME}\n({count_a} votes)', f'{MODEL_B_NAME}\n({count_b} votes)']
            percentages = [percent_a, percent_b]

            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(labels, percentages, color=['skyblue', 'lightcoral'])
            ax.set_ylabel('Percentage Judged More Empathetic (%)')
            ax.set_title(f'EMRank Comparison ({MODEL_A_NAME} vs {MODEL_B_NAME})\nEvaluated by {MODEL_NAME} (n={count_total_valid} valid evaluations)')
            ax.set_ylim(0, 100)

            # Add percentage labels on bars
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(PLOT_PNG)
            print(f"Summary plot saved to {PLOT_PNG}")
            # plt.show() # Uncomment to display plot directly if running interactively
        else:
            print("No valid judgements found to create a plot.")
    else:
        print("Results DataFrame is empty or missing 'llm_judgement' column. Skipping plot generation.")

