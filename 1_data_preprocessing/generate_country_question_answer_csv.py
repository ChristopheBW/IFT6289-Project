import pandas as pd


country_dict = {
    'CAN': 'Canada',
    'CHN': 'China',
    'GBR': 'Great Britain',
    'IDN': 'Indonesia',
}

def get_top_mode_answers_from_question(df_question, threshold=0.05):
    """
    Get the top mode answers from a question df_question.
    :param df_question: the question df, shape should be(n,) where n is the number of rows
    :param threshold: the proportional difference between these answers should be smaller than the threshold.
    :return: a dict with the top mode answers and their proportions: (list of top mode answers, list of their proportions respectively)
    """
    # get the value counts of the question
    value_counts = df_question.value_counts()
    
    # calculate the proportion of each answer
    proportions = value_counts / len(df_question)
    #print(proportions)
    
    # get the highest proportion (first value)
    max_proportion = proportions.iloc[0]
    
    # find answers where the proportion difference to the highest proportion is smaller than the threshold
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

    # convert from numpy array to list
    for i in range(len(top_mode_answers)):
        result_top_mode_answers.append(top_mode_answers[i].tolist())
        result_top_mode_answers_proportions.append(top_mode_answers_proportions[i].tolist())

    # return the top mode answers and their proportions
    return result_top_mode_answers, result_top_mode_answers_proportions

if __name__ == '__main__':
    dataset_dir = '1_data_preprocessing/dataset/culture_wvs'

    country_code = 'CHN' # Example country code

    # Construct file paths
    wvs_path = f'{dataset_dir}/{country_code}.csv'
    answer_idx_text_path = f"{dataset_dir}/manual_labeled/wvs_a_idx_text.csv"
    question_idx_text_path = f"{dataset_dir}/manual_labeled/wvs_q_idx_text.csv"

    try:
        # read the CSV files
        wvs_df = pd.read_csv(wvs_path, low_memory=False)
        answer_idx_text_df = pd.read_csv(answer_idx_text_path, low_memory=False)
        question_idx_text_df = pd.read_csv(question_idx_text_path, low_memory=False)
    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Make sure the CSV files are in the correct directory.")
        exit()


    # Filter columns starting with 'Q' followed by digits
    wvs_df_questions = wvs_df.filter(regex="^Q\d+")
    wvs_question_list = wvs_df_questions.columns.to_list()
    # --- User Defined Parameters ---
    # This proportion now applies to the SUM of the top mode answers identified by the helper function
    majority_proportion = 0.5
    # This threshold is used within the helper function to group answers with close proportions
    closeness_threshold = 0.05 # Example: Answers within 5% of the top proportion are grouped
    # ---

    results_data = [] # List to store results for the final DataFrame

    # Get the full country name from the dictionary
    country_name = country_dict.get(country_code, country_code) # Use code as fallback

    # --- Logic Implementation Start ---

    # Create sets for faster lookup of available question indices
    available_question_indices = set(question_idx_text_df['QuestionIndex'])

    # Iterate through each question column identified in the country's WVS data
    for question_index in wvs_question_list:

        # Check if this question index exists in our question text mapping file
        if question_index not in available_question_indices:
            # print(f"Skipping {question_index}: Not found in question_idx_text_df.")
            continue # Skip this question if its text definition isn't available

        # Get the response data for the current question
        df_question_responses = wvs_df_questions[question_index]

        # Calculate the top mode answer(s) and their proportions for this question
        # Pass the closeness_threshold to the helper function
        top_mode_answers, top_mode_answers_proportions = get_top_mode_answers_from_question(
            df_question_responses, threshold=closeness_threshold
        )

        # Debug print for a specific question like Q4
        # if question_index == 'Q4':
        #      print(f"\nDebug for {question_index}:")
        #      print(f"Top mode answers: {top_mode_answers}")
        #      print(f"Proportions: {top_mode_answers_proportions}")

        # Check if any top mode answers were found
        if top_mode_answers:
            # Calculate the combined proportion of ALL identified top mode answers
            combined_proportion = sum(top_mode_answers_proportions)

            # if question_index == 'Q4': # Debug print
            #     print(f"Combined proportion for {question_index}: {combined_proportion}")

            # Check if the COMBINED proportion meets the majority threshold
            if combined_proportion >= majority_proportion:
                # --- Look up Question Text (only need to do this once per question) ---
                question_text_row = question_idx_text_df[question_idx_text_df['QuestionIndex'] == question_index]
                # Should always find one because we checked `available_question_indices`
                question_text = question_text_row.iloc[0]['QuestionText']

                # --- Iterate through EACH top mode answer found ---
                for i in range(len(top_mode_answers)):
                    current_answer_index = top_mode_answers[i]
                    # current_answer_proportion = top_mode_answers_proportions[i] # Optional info

                    # --- Look up Answer Text for the current answer index ---
                    answer_text_row = answer_idx_text_df[
                        (answer_idx_text_df['QuestionIndex'] == question_index) &
                        (answer_idx_text_df['AnswerIndex'] == current_answer_index)
                    ]

                    # Check if we found a matching answer text
                    if not answer_text_row.empty:
                        current_answer_text = answer_text_row.iloc[0]['AnswerText']

                        # Append a separate result row for this specific answer
                        results_data.append({
                            'CountryText': country_name,
                            'QuestionText': question_text,
                            'AnswerText': current_answer_text
                            # Optionally add proportion: 'Proportion': current_answer_proportion
                        })
                    # else:
                    #     print(f"Warning: Answer text not found for Q:{question_index} A:{current_answer_index}. Skipping.")

    # --- Generate the CSV file ---

    # Convert the collected results into a Pandas DataFrame
    results_df = pd.DataFrame(results_data)

    # Define the output file path
    # Added closeness threshold to filename for clarity
    output_csv_path = f'{dataset_dir}/majority_answers_{country_code}_{int(majority_proportion*100)}pct_sum_{int(closeness_threshold*100)}pct_close.csv'

    # Check if any results were found before saving
    if not results_df.empty:
        # Save the DataFrame to a CSV file
        # Ensure the columns are in the desired order and exclude the DataFrame index
        results_df.to_csv(output_csv_path, index=False, columns=['CountryText', 'QuestionText', 'AnswerText'])
        print(f"Successfully generated CSV file: {output_csv_path}")
        print(f"Number of majority opinion rows found: {len(results_df)}")
    else:
        # Inform the user if no questions met the criteria
        print(f"No questions found for {country_name} where the sum of top mode answers (within {closeness_threshold*100}% proportion of max) was >= {majority_proportion*100}%.")
