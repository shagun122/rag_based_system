import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import logging
import os
import ast 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


PROCESSED_DATA_PATH = os.path.join("data", "processed_quotes.csv")
BASE_MODEL_NAME = 'all-MiniLM-L6-v2'
FINE_TUNED_MODEL_SAVE_PATH = os.path.join("models", "fine_tuned_sentence_transformer")
NUM_EPOCHS = 1 
BATCH_SIZE = 16

def load_processed_data(filepath: str) -> pd.DataFrame:
    """Loads the processed quotes data."""
    logging.info(f"Loading processed data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        df['tags'] = df['tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        logging.info(f"Loaded {len(df)} quotes.")
        return df
    except FileNotFoundError:
        logging.error(f"Error: Processed data file not found at {filepath}. Please run data_preparation.py first.")
        raise
    except Exception as e:
        logging.error(f"Error loading processed data: {e}")
        raise

def create_training_examples(df: pd.DataFrame) -> list:
    """Creates training examples for fine-tuning."""
    logging.info("Creating training examples...")
    train_examples = []
    for index, row in df.iterrows():
        quote = str(row['quote'])
        author = str(row['author'])
        tags = row['tags'] if isinstance(row['tags'], list) else []

        passage = f"Quote: {quote} Author: {author} Tags: {', '.join(tags)}"

        # Query 1: Author and first tag
        if author != "Unknown" and tags and tags[0] != "general":
            query1 = f"{author} quotes about {tags[0]}"
            train_examples.append(InputExample(texts=[query1, passage]))

        # Query 2: Just the first tag
        if tags and tags[0] != "general":
            query2 = f"quotes about {tags[0]}"
            train_examples.append(InputExample(texts=[query2, passage]))
        
        # Query 3: Just the author
        if author != "Unknown":
            query3 = f"quotes by {author}"
            train_examples.append(InputExample(texts=[query3, passage]))
        
        # Query 4: The quote itself (to improve embedding of the quote for direct search)

    logging.info(f"Created {len(train_examples)} training examples.")
    if not train_examples:
        logging.warning("No training examples were created. Check data and query generation logic.")
    return train_examples

def fine_tune_model(train_examples: list, base_model_name: str, save_path: str, epochs: int, batch_size: int):
    """Fine-tunes the sentence transformer model."""
    if not train_examples:
        logging.error("Cannot fine-tune model: No training examples provided.")
        return

    logging.info(f"Loading base model: {base_model_name}...")
    model = SentenceTransformer(base_model_name)

    logging.info("Preparing data loader and loss function...")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        logging.info(f"Created directory: {os.path.dirname(save_path)}")
        
    logging.info(f"Starting fine-tuning for {epochs} epoch(s)...")
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=epochs,
              warmup_steps=100, 
              output_path=save_path,
              show_progress_bar=True,
              checkpoint_save_steps=5000,
              checkpoint_path=save_path + "_checkpoints" 
              )
    
    logging.info(f"Fine-tuning complete. Model saved to {save_path}")

def main():
    """Main function to run the model fine-tuning pipeline."""
    logging.info("Starting model fine-tuning pipeline...")
    
    df = load_processed_data(PROCESSED_DATA_PATH)
    if df is not None and not df.empty:
        train_examples = create_training_examples(df)
        if train_examples:
            fine_tune_model(train_examples, BASE_MODEL_NAME, FINE_TUNED_MODEL_SAVE_PATH, NUM_EPOCHS, BATCH_SIZE)
        else:
            logging.error("Fine-tuning skipped due to no training examples.")
    else:
        logging.error("Fine-tuning skipped due to issues loading data.")

    logging.info("Model fine-tuning pipeline finished.")

if __name__ == "__main__":
    main()
