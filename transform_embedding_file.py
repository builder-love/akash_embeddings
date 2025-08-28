# transform_embedding_file.py
import pickle
import pandas as pd
from google.cloud import storage
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
GCS_BUCKET_NAME = "bl-repo-corpus-public"
SOURCE_PICKLE_PATH = "embeddings_data/repo_embeddings_qwen_4b.pkl"
# This is now the BASE path. A unique subfolder will be added to it.
BASE_DESTINATION_GCS_PATH = "embeddings_data/akash-qwen-checkpoints" 
BATCH_SIZE = 10000
# ---------------------

def convert_pickle_to_batched_parquet():
    """
    Downloads a large pickle file from GCS, splits it into batches,
    and uploads each batch to a unique, timestamped subfolder in GCS.
    """
    print("Starting conversion process...")

    # --- Create a unique identifier for this run ---
    # This will generate a string like "20250827-111935"
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # --- Construct the full destination path for this specific run ---
    destination_gcs_path_for_run = os.path.join(BASE_DESTINATION_GCS_PATH, run_id)
    print(f"Output for this run will be saved to: gs://{GCS_BUCKET_NAME}/{destination_gcs_path_for_run}")
    # ----------------------------------------------------
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)

    # 1. Download the large pickle file
    print(f"Downloading {SOURCE_PICKLE_PATH}...")
    blob = bucket.blob(SOURCE_PICKLE_PATH)
    pickle_data = blob.download_as_bytes()
    print("Download complete. Unpickling data...")
    embeddings_dict = pickle.loads(pickle_data)
    print(f"Unpickled {len(embeddings_dict)} total records.")

    # 2. Convert to a DataFrame to easily batch it
    df = pd.DataFrame(list(embeddings_dict.items()), columns=['repo', 'corpus_embedding'])
    del embeddings_dict # Free up memory
    
    # 3. Iterate, create batches, and upload as Parquet
    num_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Splitting into {num_batches} batches of size {BATCH_SIZE}...")

    for i in range(num_batches):
        start_index = i * BATCH_SIZE
        end_index = start_index + BATCH_SIZE
        batch_df = df.iloc[start_index:end_index]

        batch_num = i + 1
        local_filename = f"/tmp/batch_{batch_num}.parquet"
        
        # Use the new run-specific destination path
        gcs_filename = os.path.join(destination_gcs_path_for_run, f"batch_{batch_num}.parquet")

        print(f"Processing Batch {batch_num}/{num_batches}...")
        
        # Save batch to a local Parquet file
        batch_df.to_parquet(local_filename)

        # Upload the batch to GCS
        output_blob = bucket.blob(gcs_filename)
        output_blob.upload_from_filename(local_filename)
        print(f"--> Uploaded {gcs_filename}")

        # Clean up local file
        os.remove(local_filename)

    print(f"\nConversion to batched Parquet files complete! 🚀\nFiles are in: {destination_gcs_path_for_run}")

if __name__ == "__main__":
    convert_pickle_to_batched_parquet()