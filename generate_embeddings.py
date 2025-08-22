import os
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from google.cloud import storage
import logging
import numpy as np
import base64
from dotenv import load_dotenv
import psutil
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from collections import defaultdict
from tqdm import tqdm

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to generate embeddings with a chunking strategy.
    Downloads data, splits long documents into chunks, computes embeddings for each chunk,
    and uploads the results back to GCS.
    """
    logging.info("Starting embedding generation process with chunking.")
    process = psutil.Process(os.getpid())
    logging.info(f"Process ID: {process.pid}")
    logging.info(f"Initial memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    # --- 1. Configure Credentials ---
    load_dotenv()
    creds_base64 = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_BASE64")
    if creds_base64:
        creds_json_str = base64.b64decode(creds_base64).decode('utf-8')
        creds_file_path = "/tmp/gcs_creds.json"
        with open(creds_file_path, "w") as f:
            f.write(creds_json_str)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_file_path
        logging.info("Successfully configured Google Cloud credentials.")
    else:
        logging.error("GOOGLE_APPLICATION_CREDENTIALS_BASE64 environment variable is not set.")
        return

    # --- 2. Configuration ---
    gcs_bucket_name = os.environ.get("GCS_BUCKET")
    input_parquet_path = os.environ.get("INPUT_PARQUET_PATH")
    output_pickle_path = os.environ.get("OUTPUT_PICKLE_PATH")
    model_name = 'Qwen/Qwen3-Embedding-4B'
    sample_size_str = os.environ.get("SAMPLE_SIZE")

    if not all([gcs_bucket_name, input_parquet_path, output_pickle_path]):
        logging.error("Missing required environment variables.")
        return

    # --- 3. Download Data ---
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_bucket_name)
    input_blob = bucket.blob(input_parquet_path)
    local_input_path = "/tmp/input_data.parquet"
    logging.info(f"Downloading {input_parquet_path} from GCS bucket {gcs_bucket_name}...")
    input_blob.download_to_filename(local_input_path)
    logging.info("Download complete.")

    df = pd.read_parquet(local_input_path)
    logging.info(f"Loaded {len(df)} original records from Parquet file.")
    
    # --- 4. Sample Data (Optional) ---
    if sample_size_str:
        try:
            sample_size = int(sample_size_str)
            if 0 < sample_size < len(df):
                logging.info(f"Sampling {sample_size} records for testing.")
                df = df.sample(n=sample_size, random_state=42)
        except (ValueError, TypeError):
            logging.warning("Invalid SAMPLE_SIZE. Processing full dataset.")

    if 'corpus_text' not in df.columns or 'repo' not in df.columns:
        logging.error("Parquet file must contain 'corpus_text' and 'repo' columns.")
        return
        
    # --- 5. Chunking Logic ---
    logging.info("Applying chunking strategy to the corpus...")
    code_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,  # Max size of each chunk
        chunk_overlap=128 # Overlap between chunks to preserve context
    )

    chunked_data = []
    # Using tqdm for a progress bar during the chunking process
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Chunking Documents"):
        # Ensure corpus_text is a string
        text = row['corpus_text'] if isinstance(row['corpus_text'], str) else ""
        repo_name = row['repo']
        
        # Split the text into chunks
        chunks = code_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            chunked_data.append({
                'repo': repo_name,
                'chunk_id': f"{repo_name}_chunk_{i}", # Unique ID for each chunk
                'corpus_text': chunk
            })

    # Create a new DataFrame from the chunked data
    chunked_df = pd.DataFrame(chunked_data)
    logging.info(f"Original {len(df)} documents were split into {len(chunked_df)} chunks.")

    # --- 6. Load Model and Generate Embeddings ---
    logging.info(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    logging.info("Model loaded.")

    torch.cuda.empty_cache()
    logging.info("Cleared PyTorch CUDA cache.")

    corpus_chunks = chunked_df['corpus_text'].tolist()
    
    batch_size = 256
    logging.info(f"Starting embedding encoding for {len(corpus_chunks)} chunks...")

    # Use the model's built-in encode method with a progress bar
    all_embeddings = model.encode(
        corpus_chunks,
        batch_size=batch_size,
        show_progress_bar=True
    )
    logging.info("Embeddings generated successfully.")

    # --- 7. Prepare and Upload Results ---
    # Group embeddings by the original repository
    # Using defaultdict to simplify appending to lists
    repo_embeddings = defaultdict(list)
    
    repo_names = chunked_df['repo'].tolist()

    for i in range(len(repo_names)):
        repo_name = repo_names[i]
        embedding = all_embeddings[i]
        repo_embeddings[repo_name].append(embedding)

    # Convert defaultdict back to a regular dict for saving
    results = dict(repo_embeddings)
    logging.info(f"Aggregated embeddings for {len(results)} unique repositories.")
    
    local_output_path = "/tmp/repo_embeddings_chunked.pkl"
    with open(local_output_path, 'wb') as f_out:
        pickle.dump(results, f_out)
        
    output_blob = bucket.blob(output_pickle_path)
    logging.info(f"Uploading results to {output_pickle_path}...")
    output_blob.upload_from_filename(local_output_path)
    logging.info("Upload complete. Process finished.")

if __name__ == "__main__":
    main()
