# Flight Delay Prediction ML App

A professional Streamlit application for training and comparing multiple machine learning models to predict flight delays. The app loads data from Google Cloud Storage (GCS) or BigQuery and provides an interactive interface for model selection, hyperparameter tuning, and performance evaluation.

## Features

- 📥 **Flexible Data Loading**: Load data from BigQuery or directly from GCS parquet files
- 🤖 **Multiple ML Models**: Train Logistic Regression, Random Forest, and XGBoost models
- ⚙️ **Interactive Hyperparameter Tuning**: Adjust model parameters via sidebar controls
- 📊 **Comprehensive Evaluation**: View accuracy, precision, recall, F1 score, and ROC AUC metrics
- 📈 **Visualizations**: Interactive charts for metrics comparison and feature importance
- 🎯 **Feature Importance**: Visualize top features for tree-based models
- 🔍 **Confusion Matrices**: View confusion matrices for all trained models
- 🌦️ **Weather Impact Analysis**: Analyze how weather conditions affect flight delays
- 🔮 **Delay Probability Predictor**: Enter weather conditions to predict delay probability
- 🔍 **3D Clustering**: Interactive 3D visualization of flight data clusters
- ⚡ **Performance Optimized**: Automatic downsampling and caching for large datasets

## Prerequisites

1. **Python 3.8+**
2. **Google Cloud Platform (GCP) Account** with:
   - Access to the GCS bucket: `gs://team6flight/data/Flight/flights_weather_processed/ml_df`
   - BigQuery access (optional, if using BigQuery data source)
   - Appropriate authentication credentials

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /path/to/teamflight
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google Cloud authentication:**
   
   **For Local Development:**
   
   Option A: Using Application Default Credentials (Recommended)
   ```bash
   gcloud auth application-default login
   ```
   
   Option B: Set environment variable for service account
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
   ```
   
   **For Streamlit Cloud Deployment:**
   
   1. Create a service account in Google Cloud Console:
      - Go to IAM & Admin > Service Accounts
      - Create a new service account or use an existing one
      - Grant it the following roles:
        - `Storage Object Viewer` (for GCS access)
        - `BigQuery Data Viewer` (for BigQuery access)
      - Create a JSON key for the service account
   
   2. Add the service account JSON to Streamlit Cloud secrets:
      - In your Streamlit Cloud app, go to Settings > Secrets
      - Add a new secret with the key `gcp_service_account`
      - Paste the entire JSON content of your service account key file
      - The format should be:
        ```toml
        [gcp_service_account]
        type = "service_account"
        project_id = "your-project-id"
        private_key_id = "..."
        private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
        client_email = "your-service-account@project.iam.gserviceaccount.com"
        client_id = "..."
        auth_uri = "https://accounts.google.com/o/oauth2/auth"
        token_uri = "https://oauth2.googleapis.com/token"
        auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
        client_x509_cert_url = "..."
        ```
   
   3. Alternatively, you can add it as a single JSON string:
      ```toml
      [gcp_service_account]
      # Paste the entire JSON content here as a string
      ```

## Usage

### Running the App

Start the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Using the App

1. **Configure Data Source:**
   - Choose between "BigQuery" or "GCS (Direct)" in the sidebar
   - If using BigQuery, enter your GCP Project ID
   - Adjust the number of rows to load (default: 50,000, max: 300,000)

2. **Load Data:**
   - Click the "Load Data" button
   - The app will automatically detect the target column (looks for `is_delayed` or `DEP_DEL15`)
   - View data preview and summary statistics

3. **Select Models:**
   - Choose which models to train from the sidebar
   - Adjust hyperparameters for each selected model:
     - **Logistic Regression**: C (regularization), Max iterations
     - **Random Forest**: N estimators, Max depth, Min samples split/leaf
     - **XGBoost**: N estimators, Max depth, Learning rate, Subsample

4. **Train Models:**
   - Set the test set size (default: 20%)
   - Click "Train Models" button
   - Wait for training to complete (progress indicators will show)

5. **View Results:**
   - **Metrics Table**: Compare all models side-by-side
   - **Best Model**: Automatically highlighted based on F1 score
   - **Charts**: Interactive bar charts for metrics comparison
   - **Confusion Matrices**: Visual confusion matrices for each model
   - **Feature Importance**: Top 20 features for tree-based models

6. **Weather Impact Analysis:**
   - View weather impact charts (temperature, wind speed, weather events)
   - Use the **Delay Probability Predictor** to enter weather conditions
   - Get delay probability percentage based on trained models
   - See weather impact breakdown and recommendations

7. **Clustering Analysis:**
   - Enable clustering in sidebar
   - Run K-Means or DBSCAN clustering
   - View interactive 3D cluster visualization
   - Analyze cluster characteristics

## BigQuery Setup (Optional)

If you want to use BigQuery instead of direct GCS access, you'll need to create a table first.

### Creating BigQuery Table from GCS

Run this Python script to create a BigQuery table from your GCS parquet files:

```python
from google.cloud import bigquery, storage
import pandas as pd

PROJECT_ID = "your-project-id"
DATASET_ID = "flight_data"
TABLE_ID = "ml_df"
BUCKET = "team6flight"
GCS_PREFIX = "data/Flight/flights_weather_processed/ml_df"

# Initialize clients
bq_client = bigquery.Client(project=PROJECT_ID)
storage_client = storage.Client(project=PROJECT_ID)

# Create dataset if it doesn't exist
dataset_ref = bq_client.dataset(DATASET_ID)
try:
    bq_client.get_dataset(dataset_ref)
    print(f"Dataset {DATASET_ID} already exists")
except:
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = "US"
    dataset = bq_client.create_dataset(dataset, exists_ok=True)
    print(f"Created dataset {DATASET_ID}")

# Load sample data to infer schema
bucket = storage_client.bucket(BUCKET)
blobs = list(bucket.list_blobs(prefix=GCS_PREFIX))
parquet_files = [blob for blob in blobs if blob.name.endswith('.parquet')]

if parquet_files:
    gcs_uri = f"gs://{BUCKET}/{parquet_files[0].name}"
    df_sample = pd.read_parquet(gcs_uri, nrows=1000)
    
    # Create table schema from DataFrame
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True
    )
    
    # Load all parquet files
    gcs_uris = [f"gs://{BUCKET}/{blob.name}" for blob in parquet_files if blob.name.endswith('.parquet')]
    
    table_ref = dataset_ref.table(TABLE_ID)
    load_job = bq_client.load_table_from_uri(
        gcs_uris,
        table_ref,
        job_config=job_config
    )
    
    load_job.result()  # Wait for job to complete
    print(f"Loaded {load_job.output_rows} rows into {DATASET_ID}.{TABLE_ID}")
```

Or use BigQuery SQL directly:

```sql
CREATE OR REPLACE TABLE `your-project-id.flight_data.ml_df`
OPTIONS(
  format='PARQUET',
  uris=['gs://team6flight/data/Flight/flights_weather_processed/ml_df/*.parquet']
);
```

## Data Format

The app expects data with:
- **Target column**: Binary column named `is_delayed` or `DEP_DEL15` (auto-detected)
- **Features**: Numeric columns (int32, int64, float32, float64)
- **Format**: Parquet files in GCS or BigQuery table

## Model Details

### Logistic Regression
- Linear model for binary classification
- Uses L2 regularization (C parameter)
- Suitable for baseline comparisons

### Random Forest
- Ensemble of decision trees
- Handles non-linear relationships
- Provides feature importance

### XGBoost
- Gradient boosting framework
- High performance on structured data
- Built-in regularization

## Performance Considerations

- **Downsampling**: The app automatically downsamples large datasets to the specified number of rows
- **Caching**: Data loading is cached for 1 hour to avoid redundant loads
- **Feature Selection**: Only numeric features are used (categorical features can be added with preprocessing)
- **Memory Management**: Large datasets are processed in chunks when possible

## Troubleshooting

### Authentication Errors
- Ensure you have set up GCP authentication correctly
- Check that your service account has necessary permissions (Storage Object Viewer, BigQuery Data Viewer)

### Memory Issues
- Reduce the number of rows loaded
- Use fewer models or simpler hyperparameters
- Close other applications to free up memory

### Data Loading Failures
- Verify GCS bucket path is correct
- Check network connectivity
- Ensure parquet files are accessible

## License

This project is for educational and research purposes.

## Support

For issues or questions, please check:
- Streamlit documentation: https://docs.streamlit.io
- Google Cloud documentation: https://cloud.google.com/docs

