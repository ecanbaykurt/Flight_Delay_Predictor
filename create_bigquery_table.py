"""
Script to create BigQuery table from GCS parquet files
Run this script if you want to use BigQuery as the data source
"""

from google.cloud import bigquery, storage
import os

# Configuration
PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT', 'your-project-id')  # Update this
DATASET_ID = "flight_data"
TABLE_ID = "ml_df"
BUCKET = "team6flight"
GCS_PREFIX = "data/Flight/flights_weather_processed/ml_df"
LOCATION = "US"  # Change if your data is in a different region

def create_bigquery_table():
    """Create BigQuery table from GCS parquet files"""
    
    # Initialize clients
    bq_client = bigquery.Client(project=PROJECT_ID)
    storage_client = storage.Client(project=PROJECT_ID)
    
    # Create dataset if it doesn't exist
    dataset_ref = bq_client.dataset(DATASET_ID)
    try:
        bq_client.get_dataset(dataset_ref)
        print(f"✅ Dataset {DATASET_ID} already exists")
    except Exception as e:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = LOCATION
        dataset.description = "Flight delay prediction dataset"
        dataset = bq_client.create_dataset(dataset, exists_ok=True)
        print(f"✅ Created dataset {DATASET_ID}")
    
    # List all parquet files in GCS
    bucket = storage_client.bucket(BUCKET)
    blobs = list(bucket.list_blobs(prefix=GCS_PREFIX))
    parquet_files = [blob for blob in blobs if blob.name.endswith('.parquet')]
    
    if not parquet_files:
        print(f"❌ No parquet files found in gs://{BUCKET}/{GCS_PREFIX}")
        return
    
    print(f"📁 Found {len(parquet_files)} parquet files")
    
    # Create GCS URIs
    gcs_uris = [f"gs://{BUCKET}/{blob.name}" for blob in parquet_files]
    
    # Configure load job
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True  # Automatically detect schema from parquet files
    )
    
    # Load table
    table_ref = dataset_ref.table(TABLE_ID)
    print(f"🔄 Loading data into {DATASET_ID}.{TABLE_ID}...")
    print("   This may take several minutes depending on data size...")
    
    load_job = bq_client.load_table_from_uri(
        gcs_uris,
        table_ref,
        job_config=job_config
    )
    
    # Wait for job to complete
    load_job.result()
    
    # Get table info
    table = bq_client.get_table(table_ref)
    print(f"✅ Successfully loaded {table.num_rows:,} rows into {DATASET_ID}.{TABLE_ID}")
    print(f"   Table has {len(table.schema)} columns")
    print(f"   Table size: {table.num_bytes / (1024**3):.2f} GB")

if __name__ == "__main__":
    if PROJECT_ID == 'your-project-id':
        print("⚠️  Please set your PROJECT_ID:")
        print("   export GOOGLE_CLOUD_PROJECT='your-project-id'")
        print("   OR update PROJECT_ID in this script")
    else:
        create_bigquery_table()

