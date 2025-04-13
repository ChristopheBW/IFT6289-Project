import json
import os

def convert_jsonl_to_json():
    # Define input and output paths
    input_path = "1_data_preprocessing/dataset/culture_wvs/generated_dialogues/generated_dialogues_batch_CHN.jsonl"
    output_path = "1_data_preprocessing/dataset/culture_wvs/generated_dialogues/generated_dialogues_batch_CHN.json"
    
    # Ensure input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return False
    
    # Create list to hold all conversations
    all_conversations = []
    
    # Read JSONL file
    print(f"Reading from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Parse each line as JSON
                conversation = json.loads(line.strip())
                
                # Extract messages list and add to all_conversations
                if "messages" in conversation:
                    all_conversations.append(conversation["messages"])
                else:
                    print(f"Warning: Line {line_num} doesn't have 'messages' key")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
    
    # Create the final JSON structure
    output_data = {
        'messages': all_conversations
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write to output JSON file
    print(f"Writing {len(all_conversations)} conversations to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("Conversion completed successfully!")
    return True

if __name__ == "__main__":
    convert_jsonl_to_json()
