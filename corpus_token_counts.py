import os
import pandas as pd
from google.cloud import storage
import logging
from dotenv import load_dotenv
from transformers import AutoTokenizer
import psutil
from tqdm import tqdm 

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def count_tokens(text, tokenizer):
    """Counts the number of tokens in a given text using the provided tokenizer."""
    if not isinstance(text, str):
        return 0
    # Disable truncation to get the TRUE token count for long documents
    return len(tokenizer.encode(text, truncation=False))

def main():
    """
    Main function to analyze token counts in a Parquet file from GCS.
    Downloads the file, tokenizes each entry in the 'corpus_text' column,
    and prints the top 25 entries with the highest token counts.
    """
    logging.info("Starting token counting process.")
    process = psutil.Process(os.getpid())
    logging.info(f"Process ID: {process.pid}")
    logging.info(f"Initial memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    # --- 1. Load Environment Variables ---
    load_dotenv()
    logging.info("Checking for Google Cloud credentials...")

    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        logging.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not found.")
        logging.error("Please set it in your .zshrc or .env file to point to your JSON key file.")
        return
    logging.info("Google Cloud credentials found.")


    # --- 2. Download Data from GCS ---
    gcs_bucket_name = os.environ.get("GCS_BUCKET")
    input_parquet_path = os.environ.get("INPUT_PARQUET_PATH")
    model_name = 'Qwen/Qwen3-Embedding-4B'

    if not all([gcs_bucket_name, input_parquet_path]):
        logging.error("Missing one or more environment variables: GCS_BUCKET, INPUT_PARQUET_PATH")
        return

    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_bucket_name)
    input_blob = bucket.blob(input_parquet_path)
    local_input_path = "/tmp/input_data_for_token_count.parquet"
    
    logging.info(f"Downloading {input_parquet_path} from GCS bucket {gcs_bucket_name}...")
    input_blob.download_to_filename(local_input_path)
    logging.info("Download complete.")

    df = pd.read_parquet(local_input_path)
    logging.info(f"Loaded {len(df)} records from Parquet file.")
    
    if 'corpus_text' not in df.columns:
        logging.error("Parquet file must contain a 'corpus_text' column.")
        return
        
    # --- 3. Load Tokenizer and Count Tokens ---
    logging.info(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    logging.info("Tokenizer loaded.")

    logging.info("Counting tokens for each row in 'corpus_text'...")
    
    # Initialize tqdm for pandas
    tqdm.pandas(desc="Counting Tokens")
    
    # Use progress_apply to show a progress bar
    df['token_count'] = df['corpus_text'].progress_apply(lambda text: count_tokens(text, tokenizer))
    
    logging.info("Token counting complete.")

    # --- 4. Sort and Print Results ---
    df_sorted = df.sort_values(by='token_count', ascending=False)

    logging.info("Top 25 rows with the highest token counts:")
    
    pd.set_option('display.max_colwidth', 100)
    pd.set_option('display.width', 120)
    
    print("\n" + "="*80)
    print("Top 25 Longest Documents by Token Count")
    print("="*80)
    print(df_sorted[['repo', 'token_count']].head(25).to_string())
    print("="*80 + "\n")

    logging.info(f"Process finished. Final memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
