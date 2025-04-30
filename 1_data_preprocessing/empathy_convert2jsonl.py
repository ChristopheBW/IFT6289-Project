import csv
import json
import re

def convert_csv_to_jsonl(csv_filepath, jsonl_filepath):
    """Converts the Empathetic Dialogues CSV to a JSONL format suitable for Azure.

    Args:
        csv_filepath: Path to the input CSV file.
        jsonl_filepath: Path to the output JSONL file.
    """

    conversations = {}
    
    with open(csv_filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            conv_id = row['conv_id']
            if conv_id not in conversations:
                conversations[conv_id] = []
            conversations[conv_id].append(row)

    with open(jsonl_filepath, 'w', encoding='utf-8') as jsonlfile:
        for conv_id, turns in conversations.items():
            messages = []

            if turns[0]['context'] and turns[0]['prompt']:
                context = re.sub(r"[^A-Za-z0-9_\s.,;?!'\"-]", "", turns[0]['context'])
                prompt = re.sub(r"[^A-Za-z0-9_\s.,;?!'\"-]", "", turns[0]['prompt'])
                system_message = f"You are a helpful and empathetic chatbot. A user will describe a situation and their feelings. Respond in a supportive and understanding way. The user is feeling {context}. Situation: {prompt}"
                messages.append({"role": "system", "content": system_message})

            initial_speaker = turns[0]['speaker_idx']

            for i, turn in enumerate(turns):
                utterance = re.sub(r"[^A-Za-z0-9_\s.,;?!'\"-]", "", turn['utterance'])
                if not utterance.strip():
                    continue

                if turn['speaker_idx'] == initial_speaker:
                    role = "user"
                else:
                    role = "assistant"
                messages.append({"role": role, "content": utterance})
            
            if len(messages) > 1:
              jsonlfile.write(json.dumps({"messages": messages}) + '\n')


file_prefixes = ['train', 'valid', 'test']

for prefix in file_prefixes:
    csv_file = f'1_data_preprocessing/dataset/empatheticdialogues/{prefix}.csv'
    jsonl_file = f'1_data_preprocessing/dataset/empatheticdialogues/{prefix}.jsonl'
    convert_csv_to_jsonl(csv_file, jsonl_file)
    print(f"Conversion complete. {prefix} JSONL file created at: {jsonl_file}")

print("All conversions completed successfully.")