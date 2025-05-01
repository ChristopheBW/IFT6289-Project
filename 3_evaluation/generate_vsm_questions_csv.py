import csv
import os
import re

q_major_templates = {
    "1_10": "This question is about job preferences. Please think of an ideal job, disregarding your present job, if you have one. In choosing an ideal job, how important would it be to you to {question_text}. Answer with only a number where: 1 = of utmost importance, 2 = very important, 3 = of moderate importance, 4 = of little importance, 5 = of very little or no importance",
    "11_14": "In your private life, how important is {question_text} to you: Answer with only a number where: 1 = of utmost importance, 2 = very important, 3 = of moderate importance, 4 = of little importance, 5 = of very little or no importance",
    "15": "How often do you feel nervous or tense? Answer with only a number where: 1 = always, 2 = usually, 3 = sometimes, 4 = seldom, 5 = never",
    "16": "Are you a happy person? Answer with only a number where: 1 = always, 2 = usually, 3 = sometimes, 4 = seldom, 5 = never",
    "17": "Do other people or circumstances ever prevent you from doing what you really want to? Answer with only a number where: 1 = yes, always, 2 = yes, usually, 3 = sometimes, 4 = no, seldom, 5 = no, never",
    "18": "All in all, how would you describe your state of health these days? Answer with only a number where: 1 = very good, 2 = good, 3 = fair, 4 = poor, 5 = very poor",
    "19": "How proud are you to be a citizen of your country? Answer with only a number where: 1 = very proud, 2 = fairly proud, 3 = somewhat proud, 4 = not very proud, 5 = not proud at all",
    "20": "How often, in your experience, are subordinates afraid to contradict their boss (or students their teacher)? Answer with only a number where: 1 = never, 2 = seldom, 3 = sometimes, 4 = usually, 5 = always",
    "21_24": "To what extent do you agree or disagree with: {question_text}? Answer with only a number where: 1 = strongly agree, 2 = agree, 3 = undecided, 4 = disagree, 5 = strongly disagree"
}

current_dir = os.path.dirname(os.path.abspath(__file__))

input_file = os.path.join(current_dir, 'vsm_q_text.csv')
output_file = os.path.join(current_dir, 'formatted_vsm_questions.csv')

formatted_questions = []
with open(input_file, 'r') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        id_num = int(row['id'])
        
        for id_range, template in q_major_templates.items():
            range_match = re.match(r"(\d+)_(\d+)", id_range)
            single_match = re.match(r"^(\d+)$", id_range)
            
            if range_match:
                start_id, end_id = int(range_match.group(1)), int(range_match.group(2))
                if start_id <= id_num <= end_id:
                    formatted_question = template.format(question_text=row['question'])
                    formatted_questions.append({'id': id_num, 'question': formatted_question})
                    break
            elif single_match:
                single_id = int(single_match.group(1))
                if id_num == single_id:
                    formatted_questions.append({'id': id_num, 'question': template})
                    break

with open(output_file, 'w', newline='') as outfile:
    fieldnames = ['id', 'question']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(formatted_questions)

print(f"Formatted questions have been saved to {output_file}")
