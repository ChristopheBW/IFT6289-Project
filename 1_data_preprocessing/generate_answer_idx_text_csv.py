import csv
import os

# Define the general coding dictionary for special cases
general_coding = {
    -1: "Don't know",
    -2: "No answer",
    -3: "Not applicable",
    -5: "Missing"
}

# Define the standard answers for questions Q1 to Q6
q1_q6_answers = {
    1: "Very important",
    2: "Rather important",
    3: "Not very important",
    4: "Not at all important"
}

# Define the standard answers for questions Q7 to Q26
q7_q26_answers = {
    1: "Mentioned",
    2: "Not mentioned",
}

# Define the standard answers for questions Q27 to Q32
q27_q32_answers = {
    1: "Strongly agree",
    2: "Agree",
    3: "Disagree",
    4: "Strongly disagree"
}

# Define the standard answers for questions Q33 to Q41
q33_q41_answers = {
    1: "Agree strongly",
    2: "Agree",
    3: "Neither agree nor disagree",
    4: "Disagree",
    5: "Disagree strongly"
}

# Define the standard answers for questions Q42 to Q45
q42_q45_answers = {
    1: "Good",
    2: "Don't mind",
    3: "Bad",
}

# Define the standard answers for questions Q46
q46_answers = {
    1: "Very happy",
    2: "Fairly happy",
    3: "Not very happy",
    4: "Not at all happy"
}

# Define the standard answers for questions Q47
q47_answers = {
    1: "Very good",
    2: "Good",
    3: "Fair",
    4: "Poor",
    5: "Very poor"
}

# Define the standard answers for questions Q48
# q48_answers = {i: f"Freedom of choice point: {i} out of 10" for i in range(1, 11)}
# q48_answers[1] = "No choice at all"
# q48_answers[10] = "A great deal of choice"

# Define the standard answers for questions Q49 to Q50
# q49_q50_answers = {i: f"Satisfaction point: {i} out of 10" for i in range(1, 11)}
# q49_q50_answers[1] = "Completely dissatisfied"
# q49_q50_answers[10] = "Completely satisfied"

# Define the standard answers for questions Q51 to Q55
q51_q55_answers = {
    1: "Often",
    2: "Sometimes",
    3: "Rarely",
    4: "Never"
}

# Define the standard answers for questions Q56
q56_answers = {
    1: "Better off",
    2: "Worse off",
    3: "About the same"
}

# Define the standard answers for questions Q57
q57_answers = {
    1: "Most people can be trusted",
    2: "Need to be very careful"
}

# Define the standard answers for questions Q58 to Q63
q58_q63_answers = {
    1: "Trust completely",
    2: "Trust somewhat",
    3: "Do not trust very much",
    4: "Do not trust at all"
}

# Define the standard answers for questions Q64 to Q89
q64_q89_answers = {
    1: "A great deal",
    2: "Quite a lot",
    3: "Not very much",
    4: "None at all"
}

# Define the standard answers for questions Q90
# q90_answers = {i: f"Priority scale (1 being most priority on effective, 10 being most priority on democratic): {i} out of 10" for i in range(1, 11)}
# q90_answers[1] = "Being effective"
# q90_answers[10] = "Being democratic"

# Define the standard answers for questions Q91
q91_answers = {
    1: "France",
    2: "China",
    3: "India"
}

# Define the standard answers for questions Q92
q92_answers = {
    1: "Washington DC",
    2: "London",
    3: "Geneva"
}

# Define the standard answers for questions Q93
q93_answers = {
    1: "Climate change",
    2: "Human rights",
    3: "Destruction of historic monuments"
}

# Define the standard answers for questions Q94 to Q105
q94_q105_answers = {
    2: "Active member",
    1: "Inactive member",
    0: "Don't belong"
}

# Define the standard answers for questions Q106
# q106_answers = {i: f"Scale (1: Incomes should be made more equal, 10: There should be greater incentives for individual effort): {i} out of 10" for i in range(1, 11)}

# Define the standard answers for questions Q107
# q107_answers = {i: f"Scale (1: Private ownership of business and industry should be increased, 10: Government ownership of business and industry should be increased): {i} out of 10" for i in range(1, 11)}

# Define the standard answers for questions Q108
# q108_answers = {i: f"Scale (1: Government should take more responsibility to ensure that everyone is provided for, 10: People should take more responsibility to provide for themselves): {i} out of 10" for i in range(1, 11)}

# Define the standard answers for questions Q109
# q109_answers = {i: f"Scale (1: Competition is good, 10: Competition is harmful): {i} out of 10" for i in range(1, 11)}

# Define the standard answers for questions Q110
# q110_answers = {i: f"Scale (1: Hard work usually brings a better life, 10: Hard work doesn’t generally bring success): {i} out of 10" for i in range(1, 11)}

# Define the standard answers for questions Q111
q111_answers = {
    1: "Protecting the environment should be given priority, even if it causes slower economic growth and some loss of jobs.",
    2: "Economic growth and creating jobs should be the top priority, even if the environment suffers to some extent.",
    3: "Other answer (code if volunteered only!)."
}

# Define the standard answers for questions Q112
# q112_answers = {i: f"Scale (1: No corruption, 10: Abundant corruption): {i} out of 10" for i in range(1, 11)}

# Define the standard answers for questions Q113 to Q117
q113_q117_answers = {
    1: "None of them",
    2: "Few of them",
    3: "Most of them",
    4: "All of them"
}

# Define the standard answers for questions Q118
q118_answers = {
    1: "Never",
    2: "Rarely",
    3: "Frequently",
    4: "Always"
}

# Define the standard answers for questions Q119
q119_answers = {
    1: "Strongly agree",
    2: "Agree",
    3: "Disagree",
    4: "Strongly disagree",
    0: "Hard to say"
}

# Define the standard answers for questions Q120
# q120_answers = {i: f"Scale (1: No risk at all, 10: Very high risk): {i} out of 10" for i in range(1, 11)}

# Define the standard answers for questions Q121
q121_answers = {
    5: "Very good",
    4: "Quite good",
    3: "Neither good nor bad",
    2: "Quite bad",
    1: "Very bad"
}

# Define the standard answers for questions Q122 to Q129
q122_q129_answers = {
    2: "Agree",
    1: "Hard to say",
    0: "Disagree"
}

# Define the standard answers for questions Q130
q130_answers = {
    1: "Let anyone come who wants to",
    2: "Let people come as long as there are jobs available",
    3: "Place strict limits on the number of foreigners who can come here",
    4: "Prohibit people coming here from other countries"
}

# Define the standard answers for questions Q131
q131_answers = {
    1: "Very secure",
    2: "Quite secure",
    3: "Not very secure",
    4: "Not at all secure"
}

# Define the standard answers for questions Q132 to Q138
q132_q138_answers = {
    1: "Very frequently",
    2: "Quite frequently",
    3: "Not frequently",
    4: "Not at all frequently"
}

# Define the standard answers for questions Q139 to Q141
q139_q141_answers = {
    1: "Yes",
    2: "No"
}

# Define the standard answers for questions Q142 to Q143
q142_q143_answers = {
    1: "Very much",
    2: "A good deal",
    3: "Not much",
    4: "Not at all"
}

# Define the standard answers for questions Q144 to Q145
q144_q145_answers = {
    1: "Yes",
    2: "No"
}

# Define the standard answers for questions Q146 to Q148
q146_q148_answers = {
    1: "Very much",
    2: "A good deal",
    3: "Not much",
    4: "Not at all"
}

# Define the standard answers for questions Q149
q149_answers = {
    1: "Freedom",
    2: "Equality"
}

# Define the standard answers for questions Q150
q150_answers = {
    1: "Freedom",
    2: "Security"
}

# Define the standard answers for questions Q151
q151_answers = {
    1: "Yes",
    2: "No"
}

# Define the standard answers for questions Q152 to Q153
q152_q153_answers = {
    1: "A high level of economic growth",
    2: "Making sure this country has strong defense forces",
    3: "Seeing that people have more say about how things are done at their jobs and in their communities",
    4: "Trying to make our cities and countryside more beautiful"
}

# Define the standard answers for questions Q154 to Q155
q154_q155_answers = {
    1: "Maintaining order in the nation",
    2: "Giving people more say in important government decisions",
    3: "Fighting rising prices",
    4: "Protecting freedom of speech"
}

# Define the standard answers for questions Q156 to Q157
q156_q157_answers = {
    1: "A stable economy",
    2: "Progress toward a less impersonal and more humane society",
    3: "Progress toward a society in which ideas count more than money",
    4: "The fight against crime"
}

# Define the standard answers for questions Q158 to Q162
# q158_q162_answers = {i: f"Agree point: {i} out of 10" for i in range(1, 11)}
# q158_q162_answers[1] = "Completely disagree"
# q158_q162_answers[10] = "Completely agree"

# Define the standard answers for questions Q163
# q163_answers = {i: f"Scale (1: The world is a lot worse off, 10: The world is a lot better off): {i} out of 10" for i in range(1, 11)}

# Define the standard answers for questions Q164
# q164_answers = {i: f"Importance point: {i} out of 10" for i in range(1, 11)}
# q164_answers[1] = "Not at all important"
# q164_answers[10] = "Very important"

# Define the standard answers for questions Q165 to Q168
q165_q168_answers = {
    1: "Yes",
    2: "No"
}

# Define the standard answers for questions Q169 to Q170
q169_q170_answers = {
    1: "Strongly agree",
    2: "Agree",
    3: "Disagree",
    4: "Strongly disagree"
}

# Define the standard answers for questions Q173
q173_answers = {
    1: "A religious person",
    2: "Not a religious person",
    3: "An atheist"
}

# Define the standard answers for questions Q174
q174_answers = {
    1: "To follow religious norms and ceremonies",
    2: "To do good to other people"
}

# Define the standard answers for questions Q175
q175_answers = {
    1: "To make sense of life after death",
    2: "To make sense of life in this world"
}

# Define the standard answers for questions Q196 to Q198
q196_q198_answers = {
    1: "Definitely should have the right",
    2: "Probably should have the right",
    3: "Probably should not have the right",
    4: "Definitely should not have the right"
}

# Define the standard answers for questions Q199
q199_answers = {
    1: "Very interested",
    2: "Somewhat interested",
    3: "Not very interested",
    4: "Not at all interested"
}

# Define the standard answers for questions Q200
q200_answers = {
    1: "Frequently",
    2: "Occasionally",
    3: "Never"
}

# Define the standard answers for questions Q201 to Q208
q201_q208_answers = {
    1: "Daily",
    2: "Weekly",
    3: "Monthly",
    4: "Less than monthly",
    5: "Never"
}

# Define the standard answers for questions Q209 to Q220
q209_q220_answers = {
    1: "Have done",
    2: "Might do",
    3: "Would never do"
}

# Define the standard answers for questions Q221 to Q222
q221_q222_answers = {
    1: "Always",
    2: "Usually",
    3: "Never",
    4: "Not allowed to vote"
}

# Define the standard answers for questions Q224 to Q233
q224_q233_answers = {
    1: "Very often",
    2: "Fairly often",
    3: "Not often",
    4: "Not at all often"
}

# Define the standard answers for questions Q234
q234_answers = {
    1: "Very important",
    2: "Rather important",
    3: "Not very important",
    4: "Not at all important"
}

# Define the standard answers for questions Q235 to Q239
q235_q239_answers = {
    1: "Very good",
    2: "Fairly good",
    3: "Fairly bad",
    4: "Very bad"
}

# Define the standard answers for questions Q253
q253_answers = {
    1: "A great deal of respect for individual human rights",
    2: "Fairly much respect",
    3: "Not much respect",
    4: "No respect at all"
}

# Define the standard answers for questions Q254
q254_answers = {
    1: "Very proud",
    2: "Quite proud",
    3: "Not very proud",
    4: "Not at all proud",
    5: "I am not"
}

# Define the standard answers for questions Q255 to Q259
q255_q259_answers = {
    1: "Very close",
    2: "Close",
    3: "Not very close",
    4: "Not close at all"
}

if __name__ == "__main__":
    output_file = "1_data_preprocessing/dataset/culture_wvs/manual_labeled/wvs_a_idx_text.csv"
    
    # Get all question-answer dictionaries using variable name patterns
    question_mappings = []
    
    # Find all variables ending with "_answers" in the global scope
    answer_dicts = [(name, value) for name, value in globals().items() 
                    if name.endswith('_answers') and isinstance(value, dict)]
    
    for var_name, answer_dict in answer_dicts:
        # Skip commented out dictionaries
        if len(answer_dict) == 0:
            continue
            
        # Parse the variable name to extract question numbers
        q_part = var_name.replace('_answers', '')  # Remove "_answers" suffix
        
        if '_q' in q_part:
            # Handle range like q1_q6
            start_q, end_q = q_part.split('_q')
            start_num = int(start_q.replace('q', ''))
            end_num = int(end_q) + 1  # inclusive end
            question_range = range(start_num, end_num)
        else:
            # Handle single question like q46
            q_num = int(q_part.replace('q', ''))
            question_range = [q_num]
            
        question_mappings.append((question_range, answer_dict))
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['QuestionIndex', 'AnswerIndex', 'AnswerText'])
        
        # Iterate through all question ranges
        for q_range, answer_dict in question_mappings:
            for q_num in q_range:
                question_id = f"Q{q_num}"
                
                # Write specific answers for this question
                for ans_idx, ans_text in answer_dict.items():
                    writer.writerow([question_id, ans_idx, ans_text])
                
                # Write general coding answers for this question
                for ans_idx, ans_text in general_coding.items():
                    writer.writerow([question_id, ans_idx, ans_text])
    
    print(f"CSV file created at {output_file}")
