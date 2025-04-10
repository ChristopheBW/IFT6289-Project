import re
import csv
import os

# Define the file paths
md_file_path = "1_data_preprocessing/dataset/culture_wvs/manual_labeled/tmp_report.md"
csv_file_path = "1_data_preprocessing/dataset/culture_wvs/manual_labeled/wvs_q_idx_text.csv"

# Ensure the directory exists
os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

# Read the content of the markdown file
with open(md_file_path, 'r', encoding='utf-8') as md_file:
    md_content = md_file.read()

# Extract question indices and text using regex
# Modified pattern to only capture the text on the same line as the question identifier
pattern = r'\*\*(Q\d+)-\*\*\s+(.*?)(?=\n)'
matches = re.findall(pattern, md_content, re.MULTILINE)

# Write to CSV file
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['QuestionIndex', 'QuestionText'])
    for match in matches:
        q_idx, q_text = match
        writer.writerow([q_idx, q_text.strip()])

print(f"Extracted {len(matches)} questions and saved to {csv_file_path}")