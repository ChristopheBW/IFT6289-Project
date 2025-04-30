import csv
import os

def calculate_average_word_count(file_path):
    """
    Calculate the average word count for the "model_response" column in a CSV file.
    Also calculates variance, min and max word counts.
    """
    total_words = 0
    total_responses = 0
    word_counts = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                if "model_response" in row and row["model_response"]:
                    words = row["model_response"].split()
                    word_count = len(words)
                    word_counts.append(word_count)
                    total_words += word_count
                    total_responses += 1
        
        if total_responses > 0:
            average_words = total_words / total_responses
            
            # Calculate variance
            variance = sum((count - average_words) ** 2 for count in word_counts) / total_responses
            
            # Get min and max
            min_words = min(word_counts) if word_counts else 0
            max_words = max(word_counts) if word_counts else 0
            
            return average_words, total_responses, total_words, variance, min_words, max_words
        else:
            return 0, 0, 0, 0, 0, 0
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return 0, 0, 0, 0, 0, 0

def main():
    # Define file path
    csv_file_path = os.path.join("3_evaluation", "result", "empathy_llama3.2-empathy_result.csv")
    
    average_words, total_responses, total_words, variance, min_words, max_words = calculate_average_word_count(csv_file_path)
    
    if average_words > 0:
        print(f"Average word count: {average_words:.2f}")
        print(f"Variance: {variance:.2f}")
        print(f"Minimum word count: {min_words}")
        print(f"Maximum word count: {max_words}")
        print(f"Total responses analyzed: {total_responses}")
        print(f"Total word count: {total_words}")
    else:
        print("No valid responses found or error occurred.")

if __name__ == "__main__":
    main()
