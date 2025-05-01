import pandas as pd


country_dict = {
    'CAN': 'Canada',
    'CHN': 'China',
    'GBR': 'Great Britain',
    'IDN': 'Indonesia',
}

# The questions with 1-10 scale answers
q_list_1_10 = [48, 49, 50, 90, 106, 107, 108, 109, 110, 112, 120, 158, 159, 160, 161, 162, 163, 164, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252]

def get_top_mode_answers_from_question(df_question, threshold=0.05):
    """
    Get the top mode answers from a question df_question.
    
    Args:
        df_question: The question dataframe, shape should be(n,) where n is the number of rows
        threshold: The proportional difference between answers should be smaller than the threshold
        
    Returns:
        Tuple containing (list of top mode answers, list of their proportions)
    """
    value_counts = df_question.value_counts()
    proportions = value_counts / len(df_question)
    
    max_proportion = proportions.iloc[0]
    
    top_mode_answers = []
    top_mode_answers_proportions = []
    
    for i in range(len(proportions)):
        if i == 0 or (max_proportion - proportions.iloc[i]) < threshold:
            top_mode_answers.append(value_counts.index[i])
            top_mode_answers_proportions.append(proportions.iloc[i])
        else:
            break
    
    result_top_mode_answers = []
    result_top_mode_answers_proportions = []

    for i in range(len(top_mode_answers)):
        result_top_mode_answers.append(top_mode_answers[i].tolist())
        result_top_mode_answers_proportions.append(top_mode_answers_proportions[i].tolist())

    return result_top_mode_answers, result_top_mode_answers_proportions

if __name__ == '__main__':
    dataset_dir = '1_data_preprocessing/dataset/culture_wvs'

    country_code = 'IDN'

    wvs_path = f'{dataset_dir}/{country_code}.csv'
    answer_idx_text_path = f"{dataset_dir}/manual_labeled/wvs_a_idx_text.csv"
    question_idx_text_path = f"{dataset_dir}/manual_labeled/wvs_q_idx_text.csv"

    try:
        wvs_df = pd.read_csv(wvs_path, low_memory=False)
        answer_idx_text_df = pd.read_csv(answer_idx_text_path, low_memory=False)
        question_idx_text_df = pd.read_csv(question_idx_text_path, low_memory=False)
    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Make sure the CSV files are in the correct directory.")
        exit()


    wvs_df_questions = wvs_df.filter(regex="^Q\d+")
    wvs_question_list = wvs_df_questions.columns.to_list()
    
    # User Defined Parameters
    majority_proportion = 0.5
    closeness_threshold = 0.05
    
    results_data = []

    country_name = country_dict.get(country_code, country_code)

    available_question_indices = set(question_idx_text_df['QuestionIndex'])

    for question_index in wvs_question_list:

        if question_index not in available_question_indices:
            continue

        df_question_responses = wvs_df_questions[question_index]
        
        question_number = int(question_index.replace('Q', ''))
        
        if question_number in q_list_1_10:
            numeric_responses = pd.to_numeric(df_question_responses, errors='coerce')
            
            valid_responses = numeric_responses[(numeric_responses >= 1) & (numeric_responses <= 10)]
            
            total_responses = len(numeric_responses)
            valid_count = len(valid_responses)
            valid_percentage = valid_count / total_responses if total_responses > 0 else 0
            
            if valid_percentage < 0.7:
                continue
                
            average_response = valid_responses.mean()
            
            if not pd.isna(average_response):
                question_text_row = question_idx_text_df[question_idx_text_df['QuestionIndex'] == question_index]
                if not question_text_row.empty:
                    question_text = question_text_row.iloc[0]['QuestionText']
                    
                    answer_text = f"Average: {average_response:.2f} (on a scale of 1-10, {valid_percentage:.1%} valid responses)"
                    
                    results_data.append({
                        'CountryText': country_name,
                        'QuestionText': question_text,
                        'AnswerText': answer_text
                    })
        else:
            top_mode_answers, top_mode_answers_proportions = get_top_mode_answers_from_question(
                df_question_responses, threshold=closeness_threshold
            )
            
            if top_mode_answers:
                combined_proportion = sum(top_mode_answers_proportions)

                if combined_proportion >= majority_proportion:
                    question_text_row = question_idx_text_df[question_idx_text_df['QuestionIndex'] == question_index]
                    question_text = question_text_row.iloc[0]['QuestionText']

                    for i in range(len(top_mode_answers)):
                        current_answer_index = top_mode_answers[i]

                        answer_text_row = answer_idx_text_df[
                            (answer_idx_text_df['QuestionIndex'] == question_index) &
                            (answer_idx_text_df['AnswerIndex'] == current_answer_index)
                        ]

                        if not answer_text_row.empty:
                            current_answer_text = answer_text_row.iloc[0]['AnswerText']

                            results_data.append({
                                'CountryText': country_name,
                                'QuestionText': question_text,
                                'AnswerText': current_answer_text
                            })

    results_df = pd.DataFrame(results_data)

    output_csv_path = f'{dataset_dir}/majority_answers_{country_code}_{int(majority_proportion*100)}pct_sum_{int(closeness_threshold*100)}pct_close.csv'

    if not results_df.empty:
        results_df.to_csv(output_csv_path, index=False, columns=['CountryText', 'QuestionText', 'AnswerText'])
        print(f"Successfully generated CSV file: {output_csv_path}")
        print(f"Number of majority opinion rows found: {len(results_df)}")
    else:
        print(f"No questions found for {country_name} where the sum of top mode answers (within {closeness_threshold*100}% proportion of max) was >= {majority_proportion*100}%.")
