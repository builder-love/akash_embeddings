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

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to generate embeddings with a checkpointing strategy.
    For each batch of chunks, it computes embeddings and streams the results
    as a Parquet file to a GCS checkpoint directory.
    """
    logging.info("Starting embedding generation process with checkpointing.")
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
    checkpoint_gcs_path = os.environ.get("CHECKPOINT_GCS_PATH") # New path for checkpoints
    model_name = 'Qwen/Qwen3-Embedding-4B'
    sample_size_str = os.environ.get("SAMPLE_SIZE")

    if not all([gcs_bucket_name, input_parquet_path, checkpoint_gcs_path]):
        logging.error("Missing required environment variables: GCS_BUCKET, INPUT_PARQUET_PATH, CHECKPOINT_GCS_PATH")
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
        chunk_size=2048,
        chunk_overlap=128
    )

    chunked_data = []
    for index, row in df.iterrows():
        text = row['corpus_text'] if isinstance(row['corpus_text'], str) else ""
        repo_name = row['repo']
        chunks = code_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            chunked_data.append({
                'repo': repo_name,
                'chunk_id': f"{repo_name}_chunk_{i}",
                'corpus_text': chunk
            })

    chunked_df = pd.DataFrame(chunked_data)
    logging.info(f"Original {len(df)} documents were split into {len(chunked_df)} chunks.")

    # --- 6. Load and Optimize Model ---
    logging.info(f"Loading SentenceTransformer model: {model_name}")
    model_kwargs = {'torch_dtype': torch.float16}
    model = SentenceTransformer(
        model_name, 
        trust_remote_code=True,
        model_kwargs=model_kwargs
    )
    model = torch.compile(model)
    logging.info("Model loaded in float16 and compiled.")

    torch.cuda.empty_cache()
    logging.info("Cleared PyTorch CUDA cache.")

    # --- 7. Process Batches and Upload Checkpoints ---
    batch_size = 1024
    num_chunks = len(chunked_df)
    num_batches = (num_chunks + batch_size - 1) // batch_size
    
    logging.info(f"Starting embedding encoding for {num_chunks} chunks in {num_batches} batches of size {batch_size}.")

    for i in range(0, num_chunks, batch_size):
        batch_df = chunked_df.iloc[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        
        logging.info(f"Processing Batch {batch_num}/{num_batches}...")
        
        batch_corpus = batch_df['corpus_text'].tolist()
        batch_embeddings = model.encode(batch_corpus, show_progress_bar=False)
        
        # Create a DataFrame for the current batch's results
        results_df = pd.DataFrame({
            'repo': batch_df['repo'].tolist(),
            'chunk_id': batch_df['chunk_id'].tolist(),
            'embedding': list(batch_embeddings)
        })
        
        # Save batch to a local temp file
        local_output_path = f"/tmp/batch_{batch_num}.parquet"
        results_df.to_parquet(local_output_path)
        
        # Upload the batch file to GCS
        gcs_output_path = os.path.join(checkpoint_gcs_path, f"batch_{batch_num}.parquet")
        output_blob = bucket.blob(gcs_output_path)
        output_blob.upload_from_filename(local_output_path)
        
        logging.info(f"Batch {batch_num}/{num_batches} complete. Checkpoint saved to {gcs_output_path}. RAM: {process.memory_info().rss / 1024 ** 2:.2f} MB")
        
        # Clean up local temp file
        os.remove(local_output_path)

    logging.info("All batches processed and uploaded successfully. Process finished.")

if __name__ == "__main__":
    main()
