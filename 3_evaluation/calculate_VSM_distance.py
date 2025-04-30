import pandas as pd
import numpy as np

# Define real-world cultural dimension values for comparison
REAL_VALUES = {
    'PDI': {'CAN': 39, 'CHN': 80, 'IDN': 78, 'GBR': 35},
    'IDV': {'CAN': 72, 'CHN': 43, 'IDN': 5, 'GBR': 76},
    'MAS': {'CAN': 52, 'CHN': 66, 'IDN': 46, 'GBR': 66},
    'UAI': {'CAN': 48, 'CHN': 30, 'IDN': 48, 'GBR': 35},
    'LTO': {'CAN': 54, 'CHN': 77, 'IDN': 29, 'GBR': 60},
    'IVR': {'CAN': 68, 'CHN': 24, 'IDN': 38, 'GBR': 69}
}

C = 50

# Define file paths for the response data
llama_file = '3_evaluation/result/llama3.2_responses.csv'
llama_culture_file = '3_evaluation/result/llama3.2-culture_responses.csv'

def get_answer_by_id(country_data, question_id, country_code):
    """Helper function to get the answer value for a specific question ID"""
    filtered = country_data[country_data['question_id'] == question_id]
    if filtered.empty:
        print(f"Warning: No data found for question ID {question_id} in {country_code}")
        return float('nan')
    return filtered['answer'].iloc[0]  # Get the first (and should be only) answer

def get_country_data(data, country_code):
    """Filter data for a specific country and ensure answers are numeric"""
    country_data = data[data['country_code'] == country_code]
    country_data['answer'] = pd.to_numeric(country_data['answer'], errors='coerce')
    return country_data

def calculate_pdi(country_data, country_code):
    """Calculate PDI (Power Distance Index) using the formula:
    PDI = 35(m07 – m02) + 25(m20 – m23) + C(pd)
    """
    m07 = get_answer_by_id(country_data, 7, country_code)
    m02 = get_answer_by_id(country_data, 2, country_code)
    m20 = get_answer_by_id(country_data, 20, country_code)
    m23 = get_answer_by_id(country_data, 23, country_code)
    
    return 35 * (m07 - m02) + 25 * (m20 - m23) + C

def calculate_idv(country_data, country_code):
    """Calculate IDV (Individualism vs. Collectivism) using the formula:
    IDV = 35(m04 – m01) + 35(m09 – m06)
    """
    m04 = get_answer_by_id(country_data, 4, country_code)
    m01 = get_answer_by_id(country_data, 1, country_code)
    m09 = get_answer_by_id(country_data, 9, country_code)
    m06 = get_answer_by_id(country_data, 6, country_code)
    
    return 35 * (m04 - m01) + 35 * (m09 - m06) + C

def calculate_mas(country_data, country_code):
    """Calculate MAS (Masculinity vs. Femininity) using the formula:
    MAS = 35(m05 – m03) + 35(m08 – m10)
    """
    m05 = get_answer_by_id(country_data, 5, country_code)
    m03 = get_answer_by_id(country_data, 3, country_code)
    m08 = get_answer_by_id(country_data, 8, country_code)
    m10 = get_answer_by_id(country_data, 10, country_code)
    
    return 35 * (m05 - m03) + 35 * (m08 - m10) + C

def calculate_uai(country_data, country_code):
    """Calculate UAI (Uncertainty Avoidance Index) using the formula:
    UAI = 40(m18 - m15) + 25(m21 – m24)
    """
    m18 = get_answer_by_id(country_data, 18, country_code)
    m15 = get_answer_by_id(country_data, 15, country_code)
    m21 = get_answer_by_id(country_data, 21, country_code)
    m24 = get_answer_by_id(country_data, 24, country_code)
    
    return 40 * (m18 - m15) + 25 * (m21 - m24) + C

def calculate_lto(country_data, country_code):
    """Calculate LTO (Long Term Orientation) using the formula:
    LTO = 40(m13 – m14) + 25(m19 – m22)
    """
    m13 = get_answer_by_id(country_data, 13, country_code)
    m14 = get_answer_by_id(country_data, 14, country_code)
    m19 = get_answer_by_id(country_data, 19, country_code)
    m22 = get_answer_by_id(country_data, 22, country_code)
    
    return 40 * (m13 - m14) + 25 * (m19 - m22) + C

def calculate_ivr(country_data, country_code):
    """Calculate IVR (Indulgence vs. Restraint) using the formula:
    IVR = 35(m12 – m11) + 40(m17 – m16)
    """
    m12 = get_answer_by_id(country_data, 12, country_code)
    m11 = get_answer_by_id(country_data, 11, country_code)
    m17 = get_answer_by_id(country_data, 17, country_code)
    m16 = get_answer_by_id(country_data, 16, country_code)
    
    return 35 * (m12 - m11) + 40 * (m17 - m16) + C

def calculate_all_dimensions(data, country_code):
    """Calculate all cultural dimensions for a given country"""
    country_data = get_country_data(data, country_code)
    
    dimensions = {}
    try:
        dimensions['PDI'] = calculate_pdi(country_data, country_code)
        dimensions['IDV'] = calculate_idv(country_data, country_code)
        dimensions['MAS'] = calculate_mas(country_data, country_code)
        dimensions['UAI'] = calculate_uai(country_data, country_code)
        dimensions['LTO'] = calculate_lto(country_data, country_code)
        dimensions['IVR'] = calculate_ivr(country_data, country_code)
    except Exception as e:
        print(f"Error calculating dimensions for {country_code}: {e}")
    
    return dimensions

def format_value(value):
    """Format a value for display or return 'N/A' if it's NaN"""
    if np.isnan(value):
        return "N/A"
    return f"{value:.2f}"

def main():
    # Read the CSV files
    df_llama = pd.read_csv(llama_file)
    df_llama_culture = pd.read_csv(llama_culture_file)
    
    countries = ['CAN', 'CHN', 'IDN', 'GBR']
    dimensions = ['PDI', 'IDV', 'MAS', 'UAI', 'LTO', 'IVR']
    
    # Calculate all dimensions for each country in both datasets
    llama_values = {}
    llama_culture_values = {}
    
    for country in countries:
        try:
            llama_values[country] = calculate_all_dimensions(df_llama, country)
        except Exception as e:
            print(f"Error processing {country} in llama3.2: {e}")
            llama_values[country] = {dim: float('nan') for dim in dimensions}
            
        try:
            llama_culture_values[country] = calculate_all_dimensions(df_llama_culture, country)
        except Exception as e:
            print(f"Error processing {country} in llama3.2-culture: {e}")
            llama_culture_values[country] = {dim: float('nan') for dim in dimensions}
    
    # Print results for each dimension
    for dimension in dimensions:
        print(f"\n{dimension} Results:")
        print("-" * 80)
        print(f"{'Country':<10}{'Real':<15}{'llama3.2':<15}{'llama3.2-culture':<15}{'llama3.2 diff':<15}{'culture diff':<15}")
        print("-" * 80)
        
        dimension_diffs1 = []
        dimension_diffs2 = []
        
        for country in countries:
            real = REAL_VALUES[dimension][country]
            
            model1 = llama_values[country].get(dimension, float('nan'))
            model2 = llama_culture_values[country].get(dimension, float('nan'))
            
            model1_str = format_value(model1)
            model2_str = format_value(model2)
            
            if not np.isnan(model1):
                diff1 = abs(model1 - real)
                diff1_str = format_value(diff1)
                dimension_diffs1.append(diff1)
            else:
                diff1_str = "N/A"
                
            if not np.isnan(model2):
                diff2 = abs(model2 - real)
                diff2_str = format_value(diff2)
                dimension_diffs2.append(diff2)
            else:
                diff2_str = "N/A"
            
            print(f"{country:<10}{real:<15}{model1_str:<15}{model2_str:<15}{diff1_str:<15}{diff2_str:<15}")
        
        print("-" * 80)
        
        # Display average differences for this dimension
        if dimension_diffs1:
            avg_diff1 = sum(dimension_diffs1) / len(dimension_diffs1)
            print(f"Average {dimension} difference - llama3.2: {avg_diff1:.2f}")
        else:
            print(f"No valid {dimension} calculations for llama3.2")
            
        if dimension_diffs2:
            avg_diff2 = sum(dimension_diffs2) / len(dimension_diffs2)
            print(f"Average {dimension} difference - llama3.2-culture: {avg_diff2:.2f}")
        else:
            print(f"No valid {dimension} calculations for llama3.2-culture")
    
    # Calculate overall average differences across all dimensions
    print("\nOverall Average Differences:")
    print("-" * 80)
    
    all_diffs1 = []
    all_diffs2 = []
    
    for country in countries:
        for dimension in dimensions:
            real = REAL_VALUES[dimension][country]
            
            model1 = llama_values[country].get(dimension, float('nan'))
            if not np.isnan(model1):
                all_diffs1.append(abs(model1 - real))
                
            model2 = llama_culture_values[country].get(dimension, float('nan'))
            if not np.isnan(model2):
                all_diffs2.append(abs(model2 - real))
    
    if all_diffs1:
        overall_avg1 = sum(all_diffs1) / len(all_diffs1)
        print(f"Overall average difference - llama3.2: {overall_avg1:.2f}")
    else:
        print("No valid calculations for llama3.2")
        
    if all_diffs2:
        overall_avg2 = sum(all_diffs2) / len(all_diffs2)
        print(f"Overall average difference - llama3.2-culture: {overall_avg2:.2f}")
    else:
        print("No valid calculations for llama3.2-culture")

if __name__ == "__main__":
    main()
