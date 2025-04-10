import pandas as pd

def create_csv_for_country(country_code, wvs_file_path, output_dir):
    """
    Create a CSV file for a specific country code from the WVS dataset.
    Args:
        country_code (str): The country code to filter the dataset.
        wvs_df (pd.DataFrame): The DataFrame containing the WVS dataset.
    """
    wvs_df = pd.read_csv(wvs_file_path, low_memory=False)
    country_df = wvs_df[wvs_df['B_COUNTRY_ALPHA'] == country_code]
    country_df.to_csv(f'{output_dir}/{country_code}.csv', index=False)
    print(f"Created {country_code}.csv with {country_df.shape[0]} rows.")


if __name__ == "__main__":
    wvs_file_path = '1_data_preprocessing/dataset/culture_wvs/WVS_Cross-National_Wave_7_csv_v6_0.csv'
    output_dir="1_data_preprocessing/dataset/culture_wvs/"
    country_codes = ['CAN', 'CHN', 'GBR', 'IDN']

    for code in country_codes:
        create_csv_for_country(code, wvs_file_path, output_dir)
