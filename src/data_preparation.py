import pandas as pd
from datasets import load_dataset
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATASET_NAME = "Abirate/english_quotes"
PROCESSED_DATA_DIR = "data"
PROCESSED_FILENAME = "processed_quotes.csv"

def download_data(dataset_name: str):
    """Downloads the dataset from Hugging Face."""
    logging.info(f"Downloading dataset: {dataset_name}...")
    try:
        raw_dataset = load_dataset(dataset_name)
        logging.info("Dataset downloaded successfully.")
        return raw_dataset['train'].to_pandas() 
    except Exception as e:
        logging.error(f"Error downloading dataset: {e}")
        raise

def explore_data(df: pd.DataFrame):
    """Prints basic information about the DataFrame."""
    logging.info("Exploring data...")
    logging.info(f"Shape of the dataset: {df.shape}")
    logging.info(f"Columns: {df.columns.tolist()}")
    logging.info(f"First 5 rows:\n{df.head()}")
    logging.info(f"Missing values:\n{df.isnull().sum()}")

def preprocess_data(df: pd.DataFrame):
    """Preprocesses and cleans the quote data."""
    logging.info("Preprocessing data...")
    processed_df = df.copy()

    #Handle missing quotes (if any, though unlikely for 'quote' field)
    processed_df.dropna(subset=['quote'], inplace=True)

    #Clean quote text: lowercase, strip whitespace
    processed_df['quote'] = processed_df['quote'].astype(str).str.lower().str.strip()

    #Handle missing authors: fill with "Unknown"
    processed_df['author'] = processed_df['author'].fillna("Unknown").astype(str).str.strip()

    # Handle tags:
    # Tags are lists. Ensure they are lists of strings.
    # Fill missing tags with an empty list or a default tag like ['general'].
    # Lowercase and strip tags.
    def clean_tags(tags_list):
        if isinstance(tags_list, list):
            return [str(tag).lower().strip() for tag in tags_list if str(tag).strip()]
        return ['general'] 

    processed_df['tags'] = processed_df['tags'].apply(clean_tags)

   
    processed_df = processed_df[processed_df['quote'] != '']

    logging.info("Preprocessing complete.")
    logging.info(f"Shape after preprocessing: {processed_df.shape}")
    logging.info(f"First 5 rows of processed data:\n{processed_df.head()}")
    return processed_df

def save_data(df: pd.DataFrame, dir_path: str, filename: str):
    """Saves the DataFrame to a CSV file."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logging.info(f"Created directory: {dir_path}")

    filepath = os.path.join(dir_path, filename)
    try:
        df.to_csv(filepath, index=False)
        logging.info(f"Processed data saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise

def main():
    """Main function to run the data preparation pipeline."""
    logging.info("Starting data preparation pipeline...")

    df_raw = download_data(DATASET_NAME)
    explore_data(df_raw)
    df_processed = preprocess_data(df_raw)
    save_data(df_processed, PROCESSED_DATA_DIR, PROCESSED_FILENAME)

    logging.info("Data preparation pipeline finished successfully.")

if __name__ == "__main__":
    main()
