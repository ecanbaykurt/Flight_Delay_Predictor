"""
Streamlit ML App for Flight Delay Prediction
Loads data from BigQuery/GCS and trains multiple ML models
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
from google.cloud import bigquery, storage
from google.oauth2 import service_account
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    silhouette_score, davies_bouldin_score
)
# Try to import xgboost, make it optional
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    xgb = None
    XGBOOST_ERROR = str(e)
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import os
import json

# Page configuration
st.set_page_config(
    page_title="Flight Delay ML App",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced design
st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    h1 {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    h2 {
        font-size: 1.8rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    h3 {
        font-size: 1.3rem;
        margin-top: 1.5rem;
    }
    h4 {
        font-size: 1.1rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stSlider {
        padding: 0.5rem 0;
    }
    .stCheckbox {
        padding: 0.5rem 0;
    }
    .stCheckbox label {
        font-size: 1rem;
        padding: 0.5rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration
BUCKET = "team6flight"
GCS_PREFIX = "data/Flight/flights_weather_processed/ml_df"
BIGQUERY_DATASET = "flight_data"
BIGQUERY_TABLE = "ml_df"

def get_gcp_credentials():
    """Get GCP credentials from Streamlit secrets or environment"""
    # Try to get credentials from Streamlit secrets (for Streamlit Cloud)
    try:
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            try:
                # Service account JSON from Streamlit secrets
                service_account_info = st.secrets['gcp_service_account']
                if isinstance(service_account_info, str):
                    service_account_info = json.loads(service_account_info)
                credentials = service_account.Credentials.from_service_account_info(service_account_info)
                return credentials
            except Exception as e:
                # Silently fail for local development
                pass
    except Exception:
        # No secrets file available (local development)
        pass
    
    # Try environment variable for service account key file
    creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if creds_path and os.path.exists(creds_path):
        try:
            credentials = service_account.Credentials.from_service_account_file(creds_path)
            return credentials
        except Exception as e:
            st.warning(f"Error loading credentials from file: {str(e)}")
    
    # Try to use default credentials (for local development with gcloud auth)
    try:
        from google.auth import default
        credentials, _ = default()
        return credentials
    except Exception:
        pass
    
    return None


# Try to auto-detect project ID from environment or BigQuery client
def get_default_project_id():
    """Try to get project ID from environment or BigQuery client"""
    # First try environment variable
    project_id = os.environ.get('GOOGLE_CLOUD_PROJECT') or os.environ.get('GCP_PROJECT')
    
    # Try Streamlit secrets
    try:
        if not project_id and hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            try:
                service_account_info = st.secrets['gcp_service_account']
                if isinstance(service_account_info, str):
                    service_account_info = json.loads(service_account_info)
                project_id = service_account_info.get('project_id')
            except:
                pass
    except Exception:
        # No secrets file available (local development)
        pass
    
    # If not found, try to get from BigQuery client
    if not project_id:
        try:
            credentials = get_gcp_credentials()
            if credentials:
                bq_client = bigquery.Client(credentials=credentials)
                project_id = bq_client.project
        except:
            pass
    
    return project_id

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'clustering_results' not in st.session_state:
    st.session_state.clustering_results = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'numeric_features' not in st.session_state:
    st.session_state.numeric_features = None
if 'X_scaled' not in st.session_state:
    st.session_state.X_scaled = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None


@st.cache_data(ttl=3600)
def load_data_from_gcs(num_rows=50000, project_id=None):
    """Load data from GCS parquet files with downsampling"""
    try:
        # Get credentials
        credentials = get_gcp_credentials()
        
        # Use provided project_id or try to auto-detect
        if project_id:
            if credentials:
                storage_client = storage.Client(project=project_id, credentials=credentials)
            else:
                storage_client = storage.Client(project=project_id)
        else:
            # Try to auto-detect project
            detected_project = get_default_project_id()
            if detected_project:
                if credentials:
                    storage_client = storage.Client(project=detected_project, credentials=credentials)
                else:
                    storage_client = storage.Client(project=detected_project)
            else:
                # Last resort: try with credentials if available
                if credentials:
                    storage_client = storage.Client(credentials=credentials)
                else:
                    storage_client = storage.Client()
        
        bucket = storage_client.bucket(BUCKET)
        
        # List all parquet files
        blobs = list(bucket.list_blobs(prefix=GCS_PREFIX))
        parquet_files = [blob for blob in blobs if blob.name.endswith('.parquet')]
        
        if not parquet_files:
            st.error(f"No parquet files found in gs://{BUCKET}/{GCS_PREFIX}")
            return None
        
        # Read first parquet file to get structure
        first_blob = parquet_files[0]
        gcs_uri = f"gs://{BUCKET}/{first_blob.name}"
        
        # Load data with sampling
        df = pd.read_parquet(gcs_uri)
        
        # If we need more rows, load additional files
        if len(df) < num_rows and len(parquet_files) > 1:
            remaining_rows = num_rows - len(df)
            rows_per_file = remaining_rows // len(parquet_files[1:]) + 1
            
            for blob in parquet_files[1:]:
                if remaining_rows <= 0:
                    break
                gcs_uri = f"gs://{BUCKET}/{blob.name}"
                df_chunk = pd.read_parquet(gcs_uri)
                sample_size = min(rows_per_file, len(df_chunk), remaining_rows)
                if sample_size > 0:
                    df_chunk = df_chunk.sample(n=sample_size, random_state=42)
                    df = pd.concat([df, df_chunk], ignore_index=True)
                    remaining_rows -= sample_size
        
        # Downsample if needed
        if len(df) > num_rows:
            df = df.sample(n=num_rows, random_state=42).reset_index(drop=True)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data from GCS: {str(e)}")
        return None


@st.cache_data(ttl=3600)
def load_data_from_bigquery(num_rows=50000, project_id=None):
    """Load data from BigQuery with downsampling"""
    try:
        # Get credentials
        credentials = get_gcp_credentials()
        
        if project_id:
            if credentials:
                bq_client = bigquery.Client(project=project_id, credentials=credentials)
            else:
                bq_client = bigquery.Client(project=project_id)
        else:
            if credentials:
                bq_client = bigquery.Client(credentials=credentials)
            else:
                bq_client = bigquery.Client()
        
        # Check if table exists
        dataset_ref = bq_client.dataset(BIGQUERY_DATASET)
        table_ref = dataset_ref.table(BIGQUERY_TABLE)
        
        try:
            bq_client.get_table(table_ref)
        except Exception:
            st.warning(f"BigQuery table {BIGQUERY_DATASET}.{BIGQUERY_TABLE} not found. Falling back to GCS.")
            return None
        
        # Query with sampling
        query = f"""
        SELECT *
        FROM `{project_id or bq_client.project}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}`
        TABLESAMPLE SYSTEM ({min(100, (num_rows / 1000000) * 100)} PERCENT)
        LIMIT {num_rows * 2}
        """
        
        df = bq_client.query(query).to_dataframe()
        
        # Downsample if needed
        if len(df) > num_rows:
            df = df.sample(n=num_rows, random_state=42).reset_index(drop=True)
        
        return df
    
    except Exception as e:
        st.warning(f"Error loading from BigQuery: {str(e)}. Falling back to GCS.")
        return None


@st.cache_data
def detect_target_column(df):
    """Auto-detect target column - cached for performance"""
    # Check for common target column names
    target_candidates = ['is_delayed', 'DEP_DEL15', 'delayed', 'delay', 'target']
    
    for col in target_candidates:
        if col in df.columns:
            return col
    
    # If not found, look for binary columns that might be targets
    binary_cols = [col for col in df.columns 
                   if df[col].dtype in ['int64', 'int32', 'float64'] 
                   and df[col].nunique() == 2 
                   and set(df[col].dropna().unique()).issubset({0, 1})]
    
    if binary_cols:
        # Prefer columns with 'delay' in name
        delay_cols = [col for col in binary_cols if 'delay' in col.lower()]
        if delay_cols:
            return delay_cols[0]
        return binary_cols[0]
    
    return None


@st.cache_data
def get_numeric_features(df, target_col):
    """Get numeric feature columns excluding target - cached for performance"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target and other non-feature columns
    exclude_cols = [target_col, 'CANCELLED']
    numeric_features = [col for col in numeric_cols if col not in exclude_cols]
    return numeric_features


@st.cache_data
def get_airport_coordinates(df=None):
    """Get airport coordinates - from data if available, otherwise use common airports"""
    airport_coords = {}
    
    # Try to extract coordinates from data if available
    if df is not None:
        # Check for origin coordinates
        if 'ORIGIN_LAT' in df.columns and 'ORIGIN_LON' in df.columns:
            origin_coords = df[['ORIGIN', 'ORIGIN_LAT', 'ORIGIN_LON']].drop_duplicates('ORIGIN')
            for _, row in origin_coords.iterrows():
                code = str(row['ORIGIN']).strip()
                if code and code != 'nan' and pd.notna(row['ORIGIN_LAT']) and pd.notna(row['ORIGIN_LON']):
                    airport_coords[code] = (float(row['ORIGIN_LAT']), float(row['ORIGIN_LON']))
        
        # Check for destination coordinates
        if 'DEST_LAT' in df.columns and 'DEST_LON' in df.columns:
            dest_coords = df[['DEST', 'DEST_LAT', 'DEST_LON']].drop_duplicates('DEST')
            for _, row in dest_coords.iterrows():
                code = str(row['DEST']).strip()
                if code and code != 'nan' and pd.notna(row['DEST_LAT']) and pd.notna(row['DEST_LON']):
                    if code not in airport_coords:  # Don't overwrite if already exists
                        airport_coords[code] = (float(row['DEST_LAT']), float(row['DEST_LON']))
    
    # Fallback: Common US airports with coordinates (lat, lon)
    common_airports = {
        'ATL': (33.6407, -84.4277),  # Atlanta
        'LAX': (33.9425, -118.4081),  # Los Angeles
        'ORD': (41.9742, -87.9073),  # Chicago O'Hare
        'DFW': (32.8998, -97.0403),  # Dallas/Fort Worth
        'DEN': (39.8561, -104.6737),  # Denver
        'JFK': (40.6413, -73.7781),  # New York JFK
        'SFO': (37.6213, -122.3790),  # San Francisco
        'SEA': (47.4502, -122.3088),  # Seattle
        'LAS': (36.0840, -115.1537),  # Las Vegas
        'MIA': (25.7959, -80.2870),  # Miami
        'CLT': (35.2144, -80.9473),  # Charlotte
        'PHX': (33.4342, -112.0116),  # Phoenix
        'EWR': (40.6895, -74.1745),  # Newark
        'IAH': (29.9902, -95.3368),  # Houston Intercontinental
        'MCO': (28.4312, -81.3083),  # Orlando
        'MSP': (44.8831, -93.2218),  # Minneapolis
        'DTW': (42.2162, -83.3554),  # Detroit
        'PHL': (39.8719, -75.2411),  # Philadelphia
        'LGA': (40.7769, -73.8740),  # LaGuardia
        'BOS': (42.3656, -71.0096),  # Boston
        'FLL': (26.0742, -80.1506),  # Fort Lauderdale
        'IAD': (38.9531, -77.4565),  # Washington Dulles
        'DCA': (38.8512, -77.0402),  # Washington Reagan
        'SLC': (40.7899, -111.9791),  # Salt Lake City
        'MDW': (41.7868, -87.7522),  # Chicago Midway
        'HNL': (21.3206, -157.9242),  # Honolulu
        'BWI': (39.1774, -76.6684),  # Baltimore
        'DAL': (32.8471, -96.8518),  # Dallas Love
        'HOU': (29.6454, -95.2789),  # Houston Hobby
        'OAK': (37.8044, -122.2711),  # Oakland
        'SAN': (32.7338, -117.1933),  # San Diego
        'PDX': (45.5898, -122.5951),  # Portland
        'STL': (38.7487, -90.3700),  # St. Louis
        'TPA': (27.9755, -82.5332),  # Tampa
        'BNA': (36.1263, -86.6774),  # Nashville
        'AUS': (30.1945, -97.6699),  # Austin
        'ACV': (40.9781, -124.1086),  # Arcata/Eureka
        'ABE': (40.6524, -75.4404),  # Allentown/Bethlehem
        'ABI': (32.4113, -99.6819),  # Abilene Regional
        'ACT': (31.6113, -97.2305),  # Waco Regional
        'ALB': (42.7483, -73.8017),  # Albany
        'AMA': (35.2194, -101.7059),  # Amarillo
        'ANC': (61.1741, -149.9964),  # Anchorage
        'ATW': (44.2581, -88.5191),  # Appleton
        'AVL': (35.4362, -82.5418),  # Asheville
        'BDL': (41.9389, -72.6832),  # Hartford
        'BFL': (35.4336, -119.0567),  # Bakersfield
        'BGM': (42.2087, -75.9798),  # Binghamton
        'BHM': (33.5629, -86.7535),  # Birmingham
        'BIL': (45.8077, -108.5429),  # Billings
        'BIS': (46.7727, -100.7460),  # Bismarck
        'BKG': (36.5321, -93.2005),  # Branson
        'BPT': (29.9508, -94.0207),  # Beaumont
        'BQK': (31.2588, -81.4665),  # Brunswick
        'BTR': (30.5332, -91.1496),  # Baton Rouge
        'BTV': (44.4719, -73.1533),  # Burlington
        'BUF': (42.9405, -78.7322),  # Buffalo
        'BUR': (34.2006, -118.3587),  # Burbank
        'CAE': (33.9388, -81.1195),  # Columbia
        'CAK': (40.9163, -81.4422),  # Akron/Canton
        'CHS': (32.8986, -80.0405),  # Charleston
        'CID': (41.8847, -91.7108),  # Cedar Rapids
        'CLE': (41.4117, -81.8498),  # Cleveland
        'CMH': (39.9980, -82.8919),  # Columbus
        'COS': (38.8058, -104.7008),  # Colorado Springs
        'CRP': (27.7704, -97.5012),  # Corpus Christi
        'CRW': (38.3731, -81.5932),  # Charleston WV
        'CVG': (39.0488, -84.6678),  # Cincinnati
        'DAY': (39.9024, -84.2194),  # Dayton
        'DRO': (37.1515, -107.7540),  # Durango
        'DSM': (41.5340, -93.6631),  # Des Moines
        'ELP': (31.8073, -106.3776),  # El Paso
        'ERI': (42.0820, -80.1762),  # Erie
        'EUG': (44.1246, -123.2120),  # Eugene
        'EVV': (38.0367, -87.5327),  # Evansville
        'FAR': (46.9207, -96.8158),  # Fargo
        'FAT': (36.7762, -119.7181),  # Fresno
        'FCA': (48.3105, -114.2560),  # Kalispell
        'FNT': (42.9654, -83.7436),  # Flint
        'FSD': (43.5820, -96.7419),  # Sioux Falls
        'GEG': (47.6199, -117.5338),  # Spokane
        'GFK': (47.9493, -97.1761),  # Grand Forks
        'GJT': (39.1224, -108.5267),  # Grand Junction
        'GNV': (29.6901, -82.2718),  # Gainesville
        'GPT': (30.4073, -89.0701),  # Gulfport
        'GRB': (44.4851, -88.1296),  # Green Bay
        'GSO': (36.0977, -79.9373),  # Greensboro
        'GSP': (34.8957, -82.2189),  # Greenville
        'GUM': (13.4835, 144.7959),  # Guam
        'HDN': (40.4811, -107.2177),  # Hayden
        'HRL': (26.2285, -97.6544),  # Harlingen
        'HSV': (34.6372, -86.7751),  # Huntsville
        'ICT': (37.6499, -97.4331),  # Wichita
        'ILM': (34.2706, -77.9026),  # Wilmington
        'IND': (39.7173, -86.2944),  # Indianapolis
        'ISP': (40.7952, -73.1002),  # Islip
        'JAN': (32.3112, -90.0759),  # Jackson
        'JAX': (30.4941, -81.6879),  # Jacksonville
        'JNU': (58.3549, -134.5763),  # Juneau
        'KOA': (19.7388, -156.0456),  # Kona
        'LAN': (42.7787, -84.5874),  # Lansing
        'LBB': (33.6636, -101.8228),  # Lubbock
        'LEX': (38.0365, -84.6059),  # Lexington
        'LFT': (30.2053, -91.9877),  # Lafayette
        'LIT': (34.7294, -92.2243),  # Little Rock
        'MAF': (31.9425, -102.2019),  # Midland
        'MCI': (39.2976, -94.7139),  # Kansas City
        'MEM': (35.0424, -89.9767),  # Memphis
        'MFE': (26.1758, -98.2386),  # McAllen
        'MHT': (42.9326, -71.4357),  # Manchester
        'MLI': (41.4485, -90.5075),  # Moline
        'MOB': (30.6912, -88.2428),  # Mobile
        'MSN': (43.1399, -89.3375),  # Madison
        'MSY': (29.9934, -90.2581),  # New Orleans
        'MYR': (33.6797, -78.9283),  # Myrtle Beach
        'OAJ': (34.8292, -77.6121),  # Jacksonville NC
        'OKC': (35.3931, -97.6007),  # Oklahoma City
        'OMA': (41.3032, -95.8941),  # Omaha
        'ONT': (34.0560, -117.6012),  # Ontario
        'PBI': (26.6832, -80.0956),  # West Palm Beach
        'PIA': (40.6642, -89.6933),  # Peoria
        'PIT': (40.4915, -80.2329),  # Pittsburgh
        'PNS': (30.4734, -87.1866),  # Pensacola
        'PVD': (41.7326, -71.4204),  # Providence
        'RDU': (35.8776, -78.7875),  # Raleigh/Durham
        'RIC': (37.5052, -77.3197),  # Richmond
        'RNO': (39.4991, -119.7681),  # Reno
        'ROA': (37.3255, -79.9754),  # Roanoke
        'ROC': (43.1189, -77.6724),  # Rochester
        'RST': (43.9083, -92.5000),  # Rochester MN
        'RSW': (26.5362, -81.7552),  # Fort Myers
        'SAV': (32.1276, -81.2021),  # Savannah
        'SBA': (34.4262, -119.8404),  # Santa Barbara
        'SBN': (41.7087, -86.3173),  # South Bend
        'SDF': (38.1741, -85.7365),  # Louisville
        'SGF': (37.2457, -93.3886),  # Springfield MO
        'SHV': (32.4466, -93.8256),  # Shreveport
        'SJC': (37.3626, -121.9290),  # San Jose
        'SMF': (38.6954, -121.5908),  # Sacramento
        'SPI': (39.8441, -89.6779),  # Springfield IL
        'SRQ': (27.3954, -82.5544),  # Sarasota
        'SYR': (43.1112, -76.1063),  # Syracuse
        'TOL': (41.5868, -83.8078),  # Toledo
        'TUL': (36.1984, -95.8881),  # Tulsa
        'TUS': (32.1161, -110.9410),  # Tucson
        'TVC': (44.7414, -85.5822),  # Traverse City
        'TYS': (35.8110, -83.9940),  # Knoxville
        'VPS': (30.4832, -86.5254),  # Valparaiso
        'XNA': (36.2819, -94.3068),  # Fayetteville
        'YUM': (32.6566, -114.6060),  # Yuma
    }
    
    # Add common airports to the dictionary (only if not already present from data)
    for code, coords in common_airports.items():
        if code not in airport_coords:
            airport_coords[code] = coords
    
    return airport_coords


@st.cache_data
def get_airport_options(df):
    """Extract unique airports from data with codes and names"""
    airports = {}
    
    # Check for origin airports
    if 'ORIGIN' in df.columns:
        origin_data = df[['ORIGIN']].copy()
        if 'ORIGIN_CITY_NAME' in df.columns:
            origin_data['CITY'] = df['ORIGIN_CITY_NAME']
        if 'ORIGIN_STATE_NM' in df.columns:
            origin_data['STATE'] = df['ORIGIN_STATE_NM']
        
        for _, row in origin_data.drop_duplicates().iterrows():
            code = str(row['ORIGIN']).strip()
            if code and code != 'nan':
                city = str(row.get('CITY', '')).strip() if 'CITY' in row else ''
                state = str(row.get('STATE', '')).strip() if 'STATE' in row else ''
                display_name = f"{code}"
                if city:
                    display_name += f" - {city}"
                if state:
                    display_name += f", {state}"
                airports[code] = display_name
    
    # Check for destination airports
    if 'DEST' in df.columns:
        dest_data = df[['DEST']].copy()
        if 'DEST_CITY_NAME' in df.columns:
            dest_data['CITY'] = df['DEST_CITY_NAME']
        if 'DEST_STATE_NM' in df.columns:
            dest_data['STATE'] = df['DEST_STATE_NM']
        
        for _, row in dest_data.drop_duplicates().iterrows():
            code = str(row['DEST']).strip()
            if code and code != 'nan':
                city = str(row.get('CITY', '')).strip() if 'CITY' in row else ''
                state = str(row.get('STATE', '')).strip() if 'STATE' in row else ''
                display_name = f"{code}"
                if city:
                    display_name += f" - {city}"
                if state:
                    display_name += f", {state}"
                # Add to airports dict if not already there
                if code not in airports:
                    airports[code] = display_name
    
    # Sort by airport code
    sorted_airports = dict(sorted(airports.items()))
    return sorted_airports


def create_route_map(origin_code, dest_code, delay_prob=None, df=None):
    """Create an animated map showing the flight route"""
    airport_coords = get_airport_coordinates(df)
    
    # Get coordinates
    origin_coords = airport_coords.get(origin_code)
    dest_coords = airport_coords.get(dest_code)
    
    if not origin_coords or not dest_coords:
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add route line with animation
    fig.add_trace(go.Scattergeo(
        lon=[origin_coords[1], dest_coords[1]],
        lat=[origin_coords[0], dest_coords[0]],
        mode='lines',
        line=dict(width=3, color='#FF6B6B'),
        name='Route',
        showlegend=False
    ))
    
    # Add origin airport marker
    fig.add_trace(go.Scattergeo(
        lon=[origin_coords[1]],
        lat=[origin_coords[0]],
        mode='markers+text',
        marker=dict(size=15, color='#4ECDC4', symbol='circle'),
        text=[origin_code],
        textposition='top center',
        name='Origin',
        showlegend=False,
        hovertemplate=f'<b>{origin_code}</b><br>Departure Airport<extra></extra>'
    ))
    
    # Add destination airport marker
    fig.add_trace(go.Scattergeo(
        lon=[dest_coords[1]],
        lat=[dest_coords[0]],
        mode='markers+text',
        marker=dict(size=15, color='#FF6B6B', symbol='circle'),
        text=[dest_code],
        textposition='top center',
        name='Destination',
        showlegend=False,
        hovertemplate=f'<b>{dest_code}</b><br>Arrival Airport<extra></extra>'
    ))
    
    # Create animated route (great circle path)
    def great_circle_points(lat1, lon1, lat2, lon2, num_points=50):
        """Generate points along great circle route"""
        points = []
        for i in range(num_points + 1):
            frac = i / num_points
            d = math.acos(math.sin(math.radians(lat1)) * math.sin(math.radians(lat2)) +
                        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                        math.cos(math.radians(lon2 - lon1)))
            if d == 0:
                a = 1
            else:
                a = math.sin((1 - frac) * d) / math.sin(d)
            b = math.sin(frac * d) / math.sin(d)
            x = a * math.cos(math.radians(lat1)) * math.cos(math.radians(lon1)) + \
                b * math.cos(math.radians(lat2)) * math.cos(math.radians(lon2))
            y = a * math.cos(math.radians(lat1)) * math.sin(math.radians(lon1)) + \
                b * math.cos(math.radians(lat2)) * math.sin(math.radians(lon2))
            z = a * math.sin(math.radians(lat1)) + b * math.sin(math.radians(lat2))
            lat = math.degrees(math.atan2(z, math.sqrt(x * x + y * y)))
            lon = math.degrees(math.atan2(y, x))
            points.append((lat, lon))
        return points
    
    # Add animated route path
    route_points = great_circle_points(origin_coords[0], origin_coords[1], 
                                      dest_coords[0], dest_coords[1])
    route_lats = [p[0] for p in route_points]
    route_lons = [p[1] for p in route_points]
    
    # Add animated trace
    fig.add_trace(go.Scattergeo(
        lon=route_lons,
        lat=route_lats,
        mode='lines',
        line=dict(width=2, color='#95E1D3', dash='dash'),
        name='Flight Path',
        showlegend=False,
        hovertemplate='Flight Route<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Flight Route: {origin_code} → {dest_code}" + (f" (Delay Risk: {delay_prob:.1%})" if delay_prob else ""),
            x=0.5,
            font=dict(size=16)
        ),
        geo=dict(
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            coastlinecolor='rgb(204, 204, 204)',
            showocean=True,
            oceancolor='rgb(230, 245, 255)',
            showcountries=True,
            countrycolor='rgb(204, 204, 204)',
            lataxis=dict(range=[min(origin_coords[0], dest_coords[0]) - 5, 
                               max(origin_coords[0], dest_coords[0]) + 5]),
            lonaxis=dict(range=[min(origin_coords[1], dest_coords[1]) - 10, 
                               max(origin_coords[1], dest_coords[1]) + 10])
        ),
        height=400,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig


def perform_clustering(X_scaled, n_clusters, algorithm='K-Means', random_state=42):
    """Perform clustering on scaled data"""
    if algorithm == 'K-Means':
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    elif algorithm == 'DBSCAN':
        clusterer = DBSCAN(eps=0.5, min_samples=5)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    cluster_labels = clusterer.fit_predict(X_scaled)
    
    # Calculate metrics
    metrics = {}
    if algorithm == 'K-Means' and len(set(cluster_labels)) > 1:
        try:
            metrics['silhouette_score'] = silhouette_score(X_scaled, cluster_labels)
            metrics['davies_bouldin_score'] = davies_bouldin_score(X_scaled, cluster_labels)
        except:
            pass
    
    return cluster_labels, clusterer, metrics


@st.cache_data
def reduce_to_3d(X_scaled, method='PCA', random_state=42, max_samples=5000):
    """Reduce high-dimensional data to 3D for visualization - cached for performance"""
    # Sample data if too large (especially for t-SNE which is slow)
    # Reduced max_samples for better performance
    if len(X_scaled) > max_samples:
        indices = np.random.choice(len(X_scaled), max_samples, replace=False)
        X_sample = X_scaled[indices]
        sample_indices = indices
    else:
        X_sample = X_scaled
        sample_indices = np.arange(len(X_scaled))
    
    if method == 'PCA':
        reducer = PCA(n_components=3, random_state=random_state)
        X_3d = reducer.fit_transform(X_sample)
        explained_variance = reducer.explained_variance_ratio_.sum()
        return X_3d, sample_indices, {'explained_variance': explained_variance}
    
    elif method == 't-SNE':
        # Reduce perplexity and iterations for faster computation
        # Use smaller perplexity for smaller datasets
        n_samples = len(X_sample)
        perplexity = min(30, max(5, n_samples // 4))
        reducer = TSNE(n_components=3, random_state=random_state, perplexity=perplexity, n_iter=300, n_jobs=1)
        X_3d = reducer.fit_transform(X_sample)
        return X_3d, sample_indices, {}
    
    else:
        raise ValueError(f"Unknown method: {method}")


def analyze_cluster_characteristics(df_clustered, numeric_features):
    """Analyze and describe cluster characteristics"""
    cluster_descriptions = {}
    
    for cluster_id in sorted(df_clustered['cluster'].unique()):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        
        # Calculate key statistics
        stats = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df_clustered) * 100
        }
        
        # Analyze key features
        features_to_analyze = []
        if 'DEP_DEL15' in cluster_data.columns:
            delay_rate = cluster_data['DEP_DEL15'].mean()
            features_to_analyze.append(f"Delay Rate: {delay_rate:.2%}")
            stats['delay_rate'] = delay_rate
        
        if 'temp' in cluster_data.columns:
            avg_temp = cluster_data['temp'].mean()
            features_to_analyze.append(f"Avg Temp: {avg_temp:.1f}°F")
            stats['avg_temp'] = avg_temp
        
        if 'wdsp' in cluster_data.columns:
            avg_wind = cluster_data['wdsp'].mean()
            features_to_analyze.append(f"Avg Wind: {avg_wind:.1f} mph")
            stats['avg_wind'] = avg_wind
        
        if 'DEP_HOUR' in cluster_data.columns:
            avg_hour = cluster_data['DEP_HOUR'].mean()
            hour_str = f"{int(avg_hour):02d}:00"
            features_to_analyze.append(f"Avg Hour: {hour_str}")
            stats['avg_hour'] = avg_hour
        
        # Create cluster name based on characteristics
        cluster_name = f"Cluster {cluster_id}"
        if 'delay_rate' in stats:
            if stats['delay_rate'] > 0.3:
                cluster_name = f"High Delay Cluster {cluster_id}"
            elif stats['delay_rate'] < 0.1:
                cluster_name = f"Low Delay Cluster {cluster_id}"
            else:
                cluster_name = f"Medium Delay Cluster {cluster_id}"
        
        cluster_descriptions[cluster_id] = {
            'name': cluster_name,
            'stats': stats,
            'features': features_to_analyze,
            'description': f"{cluster_name}: {', '.join(features_to_analyze[:3])}"
        }
    
    return cluster_descriptions


def create_3d_cluster_plot(X_3d, cluster_labels, sample_indices, original_df, method_info=None, cluster_descriptions=None):
    """Create interactive 3D scatter plot of clusters with descriptive information"""
    # Get cluster labels for sampled data
    sampled_labels = cluster_labels[sample_indices]
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'x': X_3d[:, 0],
        'y': X_3d[:, 1],
        'z': X_3d[:, 2],
        'cluster': sampled_labels.astype(str),
        'cluster_id': sampled_labels,
        'index': sample_indices
    })
    
    # Add cluster names if available
    if cluster_descriptions:
        plot_df['cluster_name'] = plot_df['cluster_id'].map(
            lambda x: cluster_descriptions.get(x, {}).get('name', f'Cluster {x}')
        )
        plot_df['cluster_info'] = plot_df['cluster_id'].map(
            lambda x: cluster_descriptions.get(x, {}).get('description', f'Cluster {x}')
        )
    
    # Add additional info from original dataframe if available
    if original_df is not None and len(original_df) > 0:
        sampled_df = original_df.iloc[sample_indices].reset_index(drop=True)
        
        # Add all relevant columns for hover
        hover_cols = []
        for col in ['ORIGIN', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_NM', 'OP_CARRIER', 
                   'temp', 'wdsp', 'DEP_DEL15', 'DEP_HOUR', 'MONTH', 'QUARTER']:
            if col in sampled_df.columns:
                plot_df[col] = sampled_df[col].values
                hover_cols.append(col)
    
    # Create custom hover text
    hover_texts = []
    for idx, row in plot_df.iterrows():
        hover_parts = [f"<b>Cluster: {row.get('cluster_name', row['cluster'])}</b>"]
        
        if 'ORIGIN_CITY_NAME' in row and pd.notna(row['ORIGIN_CITY_NAME']):
            hover_parts.append(f"City: {row['ORIGIN_CITY_NAME']}")
        if 'ORIGIN' in row and pd.notna(row['ORIGIN']):
            hover_parts.append(f"Airport: {row['ORIGIN']}")
        if 'temp' in row and pd.notna(row['temp']):
            hover_parts.append(f"Temp: {row['temp']:.1f}°F")
        if 'wdsp' in row and pd.notna(row['wdsp']):
            hover_parts.append(f"Wind: {row['wdsp']:.1f} mph")
        if 'DEP_DEL15' in row and pd.notna(row['DEP_DEL15']):
            hover_parts.append(f"Delayed: {'Yes' if row['DEP_DEL15'] > 0.5 else 'No'}")
        if 'DEP_HOUR' in row and pd.notna(row['DEP_HOUR']):
            hover_parts.append(f"Hour: {int(row['DEP_HOUR']):02d}:00")
        
        hover_texts.append("<br>".join(hover_parts))
    
    plot_df['hover_text'] = hover_texts
    
    # Determine axis labels based on method
    if method_info and 'explained_variance' in method_info:
        x_label = f"PC1 ({method_info['explained_variance']*100:.1f}% variance)"
        y_label = "PC2"
        z_label = "PC3"
        title = f"3D Cluster Map - PCA Visualization"
    else:
        x_label = "t-SNE Dimension 1"
        y_label = "t-SNE Dimension 2"
        z_label = "t-SNE Dimension 3"
        title = f"3D Cluster Map - t-SNE Visualization"
    
    # Create 3D scatter plot
    fig = go.Figure()
    
    # Add traces for each cluster
    unique_clusters = sorted(plot_df['cluster_id'].unique())
    colors = px.colors.qualitative.Set3
    
    for i, cluster_id in enumerate(unique_clusters):
        cluster_data = plot_df[plot_df['cluster_id'] == cluster_id]
        cluster_name = cluster_data['cluster_name'].iloc[0] if 'cluster_name' in cluster_data.columns else f'Cluster {cluster_id}'
        
        fig.add_trace(go.Scatter3d(
            x=cluster_data['x'],
            y=cluster_data['y'],
            z=cluster_data['z'],
            mode='markers',
            name=cluster_name,
            marker=dict(
                size=4,
                color=colors[i % len(colors)],
                opacity=0.7,
                line=dict(width=0.5, color='white')
            ),
            text=cluster_data['hover_text'],
            hovertemplate='%{text}<extra></extra>',
            customdata=cluster_data.index
        ))
    
    # Update layout for better 3D interaction
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            bgcolor='rgb(20, 20, 20)',
            xaxis=dict(backgroundcolor='rgb(20, 20, 20)', gridcolor='gray', showbackground=True),
            yaxis=dict(backgroundcolor='rgb(20, 20, 20)', gridcolor='gray', showbackground=True),
            zaxis=dict(backgroundcolor='rgb(20, 20, 20)', gridcolor='gray', showbackground=True)
        ),
        height=800,
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    return fig


def train_models(X_train, X_test, y_train, y_test, selected_models, model_params):
    """Train selected models and return results"""
    results = {}
    
    for model_name in selected_models:
        with st.spinner(f"Training {model_name}..."):
            try:
                if model_name == "Logistic Regression":
                    model = LogisticRegression(
                        C=model_params['lr_C'],
                        max_iter=model_params['lr_max_iter'],
                        random_state=42,
                        n_jobs=-1
                    )
                
                elif model_name == "Random Forest":
                    model = RandomForestClassifier(
                        n_estimators=model_params['rf_n_estimators'],
                        max_depth=model_params['rf_max_depth'],
                        min_samples_split=model_params['rf_min_samples_split'],
                        min_samples_leaf=model_params['rf_min_samples_leaf'],
                        random_state=42,
                        n_jobs=-1
                    )
                
                elif model_name == "XGBoost":
                    if not XGBOOST_AVAILABLE:
                        results[model_name] = {'error': 'XGBoost is not available. Install libomp with "brew install libomp" to enable it.'}
                        continue
                    model = xgb.XGBClassifier(
                        n_estimators=model_params['xgb_n_estimators'],
                        max_depth=model_params['xgb_max_depth'],
                        learning_rate=model_params['xgb_learning_rate'],
                        subsample=model_params['xgb_subsample'],
                        random_state=42,
                        n_jobs=-1,
                        eval_metric='logloss'
                    )
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                roc_auc = None
                if y_pred_proba is not None:
                    try:
                        roc_auc = roc_auc_score(y_test, y_pred_proba)
                    except:
                        pass
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                # Feature importance (for tree-based models)
                feature_importance = None
                if hasattr(model, 'feature_importances_'):
                    feature_importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # For logistic regression, use absolute coefficients
                    feature_importance = np.abs(model.coef_[0])
                
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc,
                    'confusion_matrix': cm,
                    'feature_importance': feature_importance,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
            except Exception as e:
                st.error(f"Error training {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
    
    return results


def main():
    st.title("✈️ Flight Delay ML App")
    st.markdown("Simple interface for ML model training and clustering analysis")
    
    # Sidebar - Simplified
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Data source selection
        data_source = st.radio(
            "Data Source",
            ["BigQuery", "GCS (Direct)"],
            help="Choose to load from BigQuery table or directly from GCS"
        )
        
        # Auto-detect project ID
        default_project = get_default_project_id() or os.environ.get('GOOGLE_CLOUD_PROJECT', '')
        
        # Project ID input (for both BigQuery and GCS)
        project_id_input = st.text_input(
            "GCP Project ID",
            value=default_project,
            help="Your Google Cloud Project ID (required for GCS access)"
        )
        
        # Use input or fallback to auto-detected
        project_id = project_id_input.strip() if project_id_input.strip() else default_project
        
        if not project_id:
            st.warning("⚠️ Project ID is required. Please enter your GCP Project ID or set GOOGLE_CLOUD_PROJECT environment variable.")
        
        # Number of rows
        num_rows = st.slider(
            "Number of rows to load",
            min_value=10000,
            max_value=300000,
            value=50000,
            step=10000,
            help="Maximum number of rows to load (downsampled for performance). Lower values = faster performance."
        )
        
        st.divider()
        
        # Model selection
        st.header("🤖 Model Selection")
        available_models = ["Logistic Regression", "Random Forest"]
        if XGBOOST_AVAILABLE:
            available_models.append("XGBoost")
        else:
            with st.expander("ℹ️ XGBoost Not Available", expanded=False):
                st.warning("XGBoost is not available. To enable it:")
                st.code("brew install libomp", language="bash")
                st.text("Then restart the app.")
        
        selected_models = st.multiselect(
            "Select models to train",
            available_models,
            default=available_models
        )
        
        if not selected_models:
            st.warning("Please select at least one model")
        
        st.divider()
        
        # Test size
        test_size = st.slider(
            "Test set size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Proportion of data to use for testing"
        )
        
        st.divider()
        
        # Clustering section
        st.header("🔍 Clustering")
        enable_clustering = st.checkbox("Enable Clustering", value=False, help="Perform clustering analysis on the data")
        
        # Initialize clustering variables
        clustering_algorithm = "K-Means"
        n_clusters = 5
        dim_reduction_method = "PCA"
        
        if enable_clustering:
            clustering_algorithm = st.selectbox(
                "Clustering Algorithm",
                ["K-Means", "DBSCAN"],
                help="Choose clustering algorithm"
            )
            
            if clustering_algorithm == "K-Means":
                n_clusters = st.slider(
                    "Number of Clusters",
                    min_value=2,
                    max_value=20,
                    value=5,
                    step=1,
                    help="Number of clusters to create"
                )
            else:
                n_clusters = None  # DBSCAN determines clusters automatically
            
            dim_reduction_method = st.selectbox(
                "3D Visualization Method",
                ["PCA", "t-SNE"],
                help="Method to reduce dimensions to 3D for visualization. PCA is faster, t-SNE is more accurate but slower."
            )
            if dim_reduction_method == "t-SNE":
                st.caption("⚠️ Note: t-SNE is slower but provides better visualization. Consider using PCA for faster results.")
        
        st.divider()
        
        # Hyperparameters - collapsed by default
        with st.expander("⚙️ Advanced: Hyperparameters", expanded=False):
            if 'lr_C' not in st.session_state:
                st.session_state.lr_C = 1.0
                st.session_state.lr_max_iter = 200
                st.session_state.rf_n_estimators = 100
                st.session_state.rf_max_depth = 10
                st.session_state.rf_min_samples_split = 2
                st.session_state.rf_min_samples_leaf = 1
                st.session_state.xgb_n_estimators = 100
                st.session_state.xgb_max_depth = 6
                st.session_state.xgb_learning_rate = 0.1
                st.session_state.xgb_subsample = 0.8
            
            if "Logistic Regression" in selected_models:
                st.slider("LR: C", 0.01, 10.0, value=st.session_state.lr_C, step=0.1, key="lr_C")
                st.slider("LR: Max Iter", 100, 1000, value=st.session_state.lr_max_iter, step=50, key="lr_max_iter")
            if "Random Forest" in selected_models:
                st.slider("RF: N Estimators", 10, 200, value=st.session_state.rf_n_estimators, step=10, key="rf_n_estimators")
                st.slider("RF: Max Depth", 5, 30, value=st.session_state.rf_max_depth, step=1, key="rf_max_depth")
            if "XGBoost" in selected_models and XGBOOST_AVAILABLE:
                st.slider("XGB: N Estimators", 10, 200, value=st.session_state.xgb_n_estimators, step=10, key="xgb_n_estimators")
                st.slider("XGB: Max Depth", 3, 15, value=st.session_state.xgb_max_depth, step=1, key="xgb_max_depth")
                st.slider("XGB: Learning Rate", 0.01, 0.3, value=st.session_state.xgb_learning_rate, step=0.01, key="xgb_learning_rate")
    
    # Main content - Use tabs for organization
    tab1, tab2, tab3, tab4 = st.tabs(["📥 Data", "🤖 Models", "🔍 Clustering", "🌦️ Weather Impact"])
    
    # Tab 1: Data Loading
    with tab1:
        st.header("Load Data")
        
        if st.button("Load Data", type="primary") or st.session_state.data_loaded:
            if not st.session_state.data_loaded:
                with st.spinner("Loading data..."):
                    if data_source == "BigQuery" and project_id:
                        df = load_data_from_bigquery(num_rows, project_id)
                        if df is None:
                            st.info("Falling back to GCS...")
                            df = load_data_from_gcs(num_rows, project_id)
                    else:
                        df = load_data_from_gcs(num_rows, project_id)
                    
                    if df is not None and len(df) > 0:
                        st.session_state.df = df
                        st.session_state.data_loaded = True
                        st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
                    else:
                        st.error("Failed to load data. Please check your configuration.")
            else:
                df = st.session_state.df
                st.success(f"✅ Data already loaded! Shape: {df.shape}")
        
            # Display data info
            if st.session_state.data_loaded and st.session_state.df is not None:
                df = st.session_state.df
                
                # Cache target column and features to avoid recomputation
                if st.session_state.target_col is None:
                    st.session_state.target_col = detect_target_column(df)
                target_col = st.session_state.target_col
                
                if target_col is None:
                    st.error("Could not detect target column. Please ensure your data has a binary target column.")
                else:
                    st.info(f"🎯 Target: **{target_col}**")
                    
                    # Cache numeric features
                    if st.session_state.numeric_features is None:
                        st.session_state.numeric_features = get_numeric_features(df, target_col)
                    numeric_features = st.session_state.numeric_features
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", f"{len(df):,}")
                    with col2:
                        st.metric("Columns", len(df.columns))
                    with col3:
                        st.metric("Features", len(numeric_features))
                    
                    with st.expander("📋 Preview Data"):
                        st.dataframe(df.head(50))
                    
                    with st.expander("📊 Statistics"):
                        st.dataframe(df[numeric_features].describe())
    
    # Tab 2: Model Training
    with tab2:
        if not st.session_state.data_loaded:
            st.info("👈 Please load data first in the Data tab")
        else:
            df = st.session_state.df
            target_col = st.session_state.target_col or detect_target_column(df)
            numeric_features = st.session_state.numeric_features or get_numeric_features(df, target_col)
            
            # Cache these values
            if st.session_state.target_col is None:
                st.session_state.target_col = target_col
            if st.session_state.numeric_features is None:
                st.session_state.numeric_features = numeric_features
            
            st.header("Train Models")
            
            if st.button("Train Models", type="primary"):
                if not selected_models:
                    st.error("Please select at least one model from the sidebar")
                else:
                    # Check if we can reuse cached scaled data (if test_size hasn't changed)
                    if (st.session_state.X_scaled is not None and 
                        st.session_state.scaler is not None and
                        st.session_state.X_train is not None):
                        # Reuse cached data
                        X_scaled = st.session_state.X_scaled
                        scaler = st.session_state.scaler
                        X_train = st.session_state.X_train
                        X_test = st.session_state.X_test
                        y_train = st.session_state.y_train
                        y_test = st.session_state.y_test
                        st.info("♻️ Using cached preprocessed data for faster training")
                    else:
                        # Prepare features and target
                        X = df[numeric_features].fillna(0)  # Fill NaN with 0
                        y = df[target_col].fillna(0)
                        
                        # Remove any remaining NaN or inf values
                        X = X.replace([np.inf, -np.inf], 0)
                        
                        # Scale features
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        X_scaled = pd.DataFrame(X_scaled, columns=numeric_features)
                        
                        # Train-test split
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_scaled, y, test_size=test_size, random_state=42, stratify=y
                        )
                        
                        # Cache the processed data
                        st.session_state.X_scaled = X_scaled
                        st.session_state.scaler = scaler
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                    
                    st.info(f"Train set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")
                    
                    # Prepare model parameters
                    model_params = {}
                    if "Logistic Regression" in selected_models:
                        model_params.update({
                            'lr_C': st.session_state.lr_C,
                            'lr_max_iter': st.session_state.lr_max_iter
                        })
                    if "Random Forest" in selected_models:
                        model_params.update({
                            'rf_n_estimators': st.session_state.rf_n_estimators,
                            'rf_max_depth': st.session_state.rf_max_depth,
                            'rf_min_samples_split': st.session_state.rf_min_samples_split,
                            'rf_min_samples_leaf': st.session_state.rf_min_samples_leaf
                        })
                    if "XGBoost" in selected_models and XGBOOST_AVAILABLE:
                        model_params.update({
                            'xgb_n_estimators': st.session_state.xgb_n_estimators,
                            'xgb_max_depth': st.session_state.xgb_max_depth,
                            'xgb_learning_rate': st.session_state.xgb_learning_rate,
                            'xgb_subsample': st.session_state.xgb_subsample
                        })
                    
                    # Train models
                    results = train_models(X_train, X_test, y_train, y_test, selected_models, model_params)
                    st.session_state.results = results
                    st.session_state.models_trained = True
                    st.session_state.feature_names = numeric_features
                    st.session_state.scaler = scaler
        
            # Display results
            if st.session_state.models_trained and st.session_state.results:
                st.header("Results")
                
                results = st.session_state.results
                feature_names = st.session_state.get('feature_names', [])
                
                # Create results dataframe
                results_data = []
                for model_name, result in results.items():
                    if 'error' not in result:
                        results_data.append({
                            'Model': model_name,
                            'Accuracy': result['accuracy'],
                            'Precision': result['precision'],
                            'Recall': result['recall'],
                            'F1 Score': result['f1'],
                            'ROC AUC': result['roc_auc'] if result['roc_auc'] else 'N/A'
                        })
                
                if results_data:
                    results_df = pd.DataFrame(results_data)
                    
                    # Find best model
                    best_model_idx = results_df['F1 Score'].idxmax()
                    best_model = results_df.loc[best_model_idx, 'Model']
                    st.success(f"🏆 Best: **{best_model}** (F1: {results_df.loc[best_model_idx, 'F1 Score']:.4f})")
                    
                    # Display results table
                    st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1 Score']))
                    
                    # Metrics visualization
                    st.subheader("Metrics Chart")
                    
                    # Bar chart for metrics
                    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                    fig = go.Figure()
                    
                    for metric in metrics_to_plot:
                        fig.add_trace(go.Bar(
                            name=metric,
                            x=results_df['Model'],
                            y=results_df[metric],
                            text=[f'{v:.3f}' for v in results_df[metric]],
                            textposition='auto'
                        ))
                    
                    fig.update_layout(
                        title="Model Performance Metrics",
                        xaxis_title="Model",
                        yaxis_title="Score",
                        barmode='group',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Confusion matrices - collapsed
                    with st.expander("🔍 Confusion Matrices", expanded=False):
                        cols = st.columns(len(results))
                        for idx, (model_name, result) in enumerate(results.items()):
                            if 'error' not in result:
                                with cols[idx]:
                                    cm = result['confusion_matrix']
                                    fig_cm = px.imshow(cm, labels=dict(x="Predicted", y="Actual"), 
                                                      x=['No Delay', 'Delay'], y=['No Delay', 'Delay'],
                                                      text_auto=True, title=model_name)
                                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                    # Feature importance - collapsed
                    with st.expander("🎯 Feature Importance", expanded=False):
                        for model_name, result in results.items():
                            if 'error' not in result and result.get('feature_importance') is not None and feature_names:
                                importance = result['feature_importance']
                                importance_df = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Importance': importance
                                }).sort_values('Importance', ascending=False).head(15)
                                
                                fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                                               title=f"{model_name}")
                                fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                                st.plotly_chart(fig_imp, use_container_width=True)
    
    # Tab 3: Clustering
    with tab3:
        if not st.session_state.data_loaded:
            st.info("👈 Please load data first in the Data tab")
        elif not enable_clustering:
            st.info("👈 Enable clustering in the sidebar first")
        else:
        
            st.header("Clustering Analysis")
            
            df = st.session_state.df
            target_col = st.session_state.target_col or detect_target_column(df)
            numeric_features = st.session_state.numeric_features or get_numeric_features(df, target_col)
            
            if st.button("Run Clustering", type="primary"):
                with st.spinner("Performing clustering analysis..."):
                    # Reuse cached scaled data if available (from model training)
                    if st.session_state.X_scaled is not None and len(st.session_state.X_scaled) == len(df):
                        X_scaled = st.session_state.X_scaled.values if isinstance(st.session_state.X_scaled, pd.DataFrame) else st.session_state.X_scaled
                        st.info("♻️ Using cached scaled data for faster clustering")
                    else:
                        # Prepare data
                        X = df[numeric_features].fillna(0)
                        X = X.replace([np.inf, -np.inf], 0)
                        
                        # Scale features
                        scaler_cluster = StandardScaler()
                        X_scaled = scaler_cluster.fit_transform(X)
                    
                    # Perform clustering
                    cluster_labels, clusterer, metrics = perform_clustering(
                        X_scaled, 
                        n_clusters if clustering_algorithm == "K-Means" else None,
                        clustering_algorithm
                    )
                    
                    # Add cluster labels to dataframe
                    df_with_clusters = df.copy()
                    df_with_clusters['cluster'] = cluster_labels
                    
                    # Reduce to 3D for visualization (cached function)
                    with st.spinner("Reducing dimensions to 3D for visualization (this may take a moment for t-SNE)..."):
                        # Convert to numpy array if needed for caching
                        X_scaled_array = X_scaled.values if isinstance(X_scaled, pd.DataFrame) else X_scaled
                        X_3d, sample_indices, method_info = reduce_to_3d(
                            X_scaled_array, 
                            method=dim_reduction_method,
                            random_state=42
                        )
                    
                    # Analyze cluster characteristics
                    cluster_descriptions = analyze_cluster_characteristics(df_with_clusters, numeric_features)
                    
                    # Store results
                    st.session_state.clustering_results = {
                        'cluster_labels': cluster_labels,
                        'clusterer': clusterer,
                        'metrics': metrics,
                        'X_scaled': X_scaled,
                        'X_3d': X_3d,
                        'sample_indices': sample_indices,
                        'method_info': method_info,
                        'df_with_clusters': df_with_clusters,
                        'dim_reduction_method': dim_reduction_method,
                        'cluster_descriptions': cluster_descriptions,
                        'numeric_features': numeric_features
                    }
            
            # Display clustering results
            if st.session_state.clustering_results is not None:
                results = st.session_state.clustering_results
                
                # Show metrics
                if results['metrics']:
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'silhouette_score' in results['metrics']:
                            st.metric("Silhouette Score", f"{results['metrics']['silhouette_score']:.4f}")
                    with col2:
                        if 'davies_bouldin_score' in results['metrics']:
                            st.metric("Davies-Bouldin Score", f"{results['metrics']['davies_bouldin_score']:.4f}")
                
                # Cluster distribution - simplified
                st.subheader("Cluster Distribution")
                cluster_counts = pd.Series(results['cluster_labels']).value_counts().sort_index()
                
                # Use cluster names if available
                if 'cluster_descriptions' in results and results['cluster_descriptions']:
                    cluster_names = [results['cluster_descriptions'].get(cid, {}).get('name', f'Cluster {cid}') 
                                   for cid in cluster_counts.index]
                    cluster_df = pd.DataFrame({
                        'Cluster Name': cluster_names,
                        'Cluster ID': cluster_counts.index.astype(str),
                        'Count': cluster_counts.values,
                        'Percentage': (cluster_counts.values / len(results['cluster_labels']) * 100).round(2)
                    })
                else:
                    cluster_df = pd.DataFrame({
                        'Cluster': cluster_counts.index.astype(str),
                        'Count': cluster_counts.values,
                        'Percentage': (cluster_counts.values / len(results['cluster_labels']) * 100).round(2)
                    })
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(cluster_df, use_container_width=True)
                with col2:
                    x_col = 'Cluster Name' if 'Cluster Name' in cluster_df.columns else 'Cluster'
                    fig_dist = px.bar(
                        cluster_df,
                        x=x_col,
                        y='Count',
                        title='Number of Points per Cluster',
                        color=x_col,
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_dist.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                # 3D Visualization - Main focus
                st.subheader("3D Cluster Map")
                st.caption("Rotate: drag | Zoom: scroll | Pan: Shift+drag | Hover: see details")
                
                fig_3d = create_3d_cluster_plot(
                    results['X_3d'],
                    results['cluster_labels'],
                    results['sample_indices'],
                    results['df_with_clusters'],
                    results['method_info'],
                    results.get('cluster_descriptions')
                )
                st.plotly_chart(fig_3d, use_container_width=True, height=800)
                
                # Cluster descriptions - collapsed
                if 'cluster_descriptions' in results and results['cluster_descriptions']:
                    with st.expander("📋 Cluster Descriptions", expanded=False):
                        desc_cols = st.columns(min(3, len(results['cluster_descriptions'])))
                        for idx, (cluster_id, desc) in enumerate(sorted(results['cluster_descriptions'].items())):
                            with desc_cols[idx % len(desc_cols)]:
                                st.markdown(f"**{desc['name']}**")
                                st.write(f"Size: {desc['stats']['size']:,} ({desc['stats']['percentage']:.1f}%)")
                                for feature in desc['features'][:2]:
                                    st.caption(feature)
                
                # Cluster statistics - collapsed
                with st.expander("🔬 Cluster Statistics", expanded=False):
                    df_clustered = results['df_with_clusters']
                    cluster_stats = []
                    for cluster_id in sorted(df_clustered['cluster'].unique()):
                        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
                        
                        # Get cluster name
                        cluster_name = f'Cluster {cluster_id}'
                        if 'cluster_descriptions' in results and results['cluster_descriptions']:
                            cluster_name = results['cluster_descriptions'].get(cluster_id, {}).get('name', cluster_name)
                        
                        stats = {
                            'Cluster Name': cluster_name,
                            'Cluster ID': cluster_id,
                            'Size': len(cluster_data),
                            'Percentage': f"{(len(cluster_data) / len(df_clustered) * 100):.2f}%"
                        }
                        
                        # Add mean values for key numeric features
                        key_features = ['DEP_DEL15', 'temp', 'wdsp', 'DEP_HOUR', 'visib']
                        for feature in key_features:
                            if feature in cluster_data.columns:
                                mean_val = cluster_data[feature].mean()
                                if feature == 'DEP_DEL15':
                                    stats['Delay Rate'] = f"{mean_val:.2%}"
                                elif feature == 'DEP_HOUR':
                                    stats['Avg Hour'] = f"{int(mean_val):02d}:00"
                                else:
                                    stats[f'Avg {feature}'] = f"{mean_val:.2f}"
                        
                        cluster_stats.append(stats)
                    
                    stats_df = pd.DataFrame(cluster_stats)
                    st.dataframe(stats_df, use_container_width=True)
                
                # Download clustered data
                st.subheader("💾 Download Results")
                csv = df_clustered.to_csv(index=False)
                st.download_button(
                    label="Download Clustered Data (CSV)",
                    data=csv,
                    file_name="clustered_flight_data.csv",
                    mime="text/csv"
                )
    
    # Tab 4: Weather Impact Analysis & Delay Predictor
    with tab4:
        st.header("🌦️ Weather Impact on Flight Delays")
        st.markdown("**Enter weather conditions to predict delay probability**")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data first in the Data tab")
        elif not st.session_state.models_trained:
            st.warning("⚠️ Please train models first in the Models tab")
        else:
            df = st.session_state.df
            target_col = st.session_state.target_col or detect_target_column(df)
            numeric_features = st.session_state.numeric_features or get_numeric_features(df, target_col)
            
            # Weather Impact Analysis Section
            st.subheader("📊 Weather Impact Analysis")
            
            # Analyze weather effects from real data
            if 'temp' in df.columns and 'DEP_DEL15' in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Temperature Impact**")
                    temp_bins = pd.cut(df['temp'], bins=5, labels=['Very Cold', 'Cold', 'Moderate', 'Warm', 'Hot'])
                    temp_delay = df.groupby(temp_bins)['DEP_DEL15'].mean() * 100
                    fig_temp = px.bar(x=temp_delay.index.astype(str), y=temp_delay.values,
                                     title="Delay Rate by Temperature",
                                     labels={'x': 'Temperature Range', 'y': 'Delay Rate (%)'})
                    st.plotly_chart(fig_temp, use_container_width=True)
                
                with col2:
                    st.write("**Wind Speed Impact**")
                    wind_bins = pd.cut(df['wdsp'], bins=5, labels=['Calm', 'Light', 'Moderate', 'Strong', 'Very Strong'])
                    wind_delay = df.groupby(wind_bins)['DEP_DEL15'].mean() * 100
                    fig_wind = px.bar(x=wind_delay.index.astype(str), y=wind_delay.values,
                                     title="Delay Rate by Wind Speed",
                                     labels={'x': 'Wind Speed Range', 'y': 'Delay Rate (%)'})
                    st.plotly_chart(fig_wind, use_container_width=True)
            
            # Weather event impact
            weather_events = ['fog', 'rain_drizzle', 'snow_ice_pellets', 'hail', 'thunder', 'tornado_funnel_cloud']
            available_events = [e for e in weather_events if e in df.columns]
            
            if available_events:
                st.write("**Weather Event Impact**")
                event_delays = {}
                for event in available_events:
                    event_delays[event.replace('_', ' ').title()] = df.groupby(event)['DEP_DEL15'].mean() * 100
                
                event_df = pd.DataFrame(event_delays).T
                event_df.columns = ['No Event', 'With Event']
                fig_events = px.bar(event_df, barmode='group', 
                                   title="Delay Rate: With vs Without Weather Events",
                                   labels={'value': 'Delay Rate (%)', 'index': 'Weather Event'})
                st.plotly_chart(fig_events, use_container_width=True)
            
            st.divider()
            
            # Delay Predictor Section - Minimal Design with Icons
            st.markdown("---")
            st.markdown("### 🌤️ Delay Probability Predictor")
            st.caption("Enter weather conditions and airports to predict flight delay probability")
            
            # Get airport options from data
            airport_options = get_airport_options(df)
            airport_codes = list(airport_options.keys()) if airport_options else []
            airport_display = list(airport_options.values()) if airport_options else []
            
            # Get default values from data
            default_temp = df['temp'].median() if 'temp' in df.columns else 60
            default_wind = df['wdsp'].median() if 'wdsp' in df.columns else 10
            default_hour = df['DEP_HOUR'].median() if 'DEP_HOUR' in df.columns else 12
            
            # Get default airports from data
            default_origin = None
            default_dest = None
            if 'ORIGIN' in df.columns and len(df['ORIGIN'].mode()) > 0:
                default_origin = df['ORIGIN'].mode()[0]
            if 'DEST' in df.columns and len(df['DEST'].mode()) > 0:
                default_dest = df['DEST'].mode()[0]
            
            # Minimal Input form with descriptive icons
            with st.form("weather_prediction_form"):
                # Airport Selection
                origin_code = None
                dest_code = None
                if airport_codes and len(airport_codes) > 0:
                    st.markdown("#### ✈️ Flight Route")
                    airport_col1, airport_col2 = st.columns(2)
                    
                    with airport_col1:
                        origin_idx = 0
                        if default_origin and default_origin in airport_codes:
                            origin_idx = airport_codes.index(default_origin)
                        origin_display = st.selectbox(
                            "🛫 Departure Airport",
                            options=airport_display,
                            index=origin_idx if origin_idx < len(airport_display) else 0,
                            help="Select your departure airport"
                        )
                        origin_code = airport_codes[airport_display.index(origin_display)]
                    
                    with airport_col2:
                        dest_idx = 0
                        if default_dest and default_dest in airport_codes:
                            dest_idx = airport_codes.index(default_dest)
                        dest_display = st.selectbox(
                            "🛬 Arrival Airport",
                            options=airport_display,
                            index=dest_idx if dest_idx < len(airport_display) else 0,
                            help="Select your arrival airport"
                        )
                        dest_code = airport_codes[airport_display.index(dest_display)]
                    
                    st.markdown("---")
                
                # Weather Conditions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    temp = st.slider("🌡️ Temperature (°F)", -50.0, 120.0, float(default_temp), 1.0)
                    if 'visib' in numeric_features:
                        visib = st.slider("👁️ Visibility (miles)", 0.0, 20.0, 10.0, 0.1)
                    else:
                        visib = None
                
                with col2:
                    wdsp = st.slider("💨 Wind Speed (mph)", 0.0, 100.0, float(default_wind), 1.0)
                    if 'dewp' in numeric_features:
                        dewp = st.slider("💧 Dew Point (°F)", -50.0, 100.0, 
                                        float(df['dewp'].median()) if 'dewp' in df.columns else 50.0, 1.0)
                    else:
                        dewp = None
                
                with col3:
                    dep_hour = st.slider("🕐 Departure Hour", 0, 23, int(default_hour))
                
                st.markdown("---")
                
                # Weather Events with descriptive icons
                col4, col5, col6 = st.columns(3)
                
                with col4:
                    fog = st.checkbox("🌫️ Fog", value=False) if 'fog' in numeric_features else False
                    rain = st.checkbox("🌧️ Rain/Drizzle", value=False) if 'rain_drizzle' in numeric_features else False
                
                with col5:
                    snow = st.checkbox("❄️ Snow/Ice", value=False) if 'snow_ice_pellets' in numeric_features else False
                    hail = st.checkbox("🧊 Hail", value=False) if 'hail' in numeric_features else False
                
                with col6:
                    thunder = st.checkbox("⚡ Thunder", value=False) if 'thunder' in numeric_features else False
                    tornado = st.checkbox("🌪️ Tornado/Funnel Cloud", value=False) if 'tornado_funnel_cloud' in numeric_features else False
                
                predict_button = st.form_submit_button("🔮 Predict Delay Probability", type="primary", use_container_width=True)
            
            # Make prediction
            if predict_button:
                if not st.session_state.results:
                    st.error("No trained models available. Please train models first.")
                else:
                    # Prepare input data
                    input_data = {}
                    
                    # Handle airport codes if airports were selected
                    if origin_code and dest_code:
                        # Check if ORIGIN/DEST are in numeric features (might be encoded)
                        # If they're categorical, we'll use the mode or a default
                        if 'ORIGIN' in df.columns:
                            # Use the selected origin airport's typical values if available
                            origin_data = df[df['ORIGIN'] == origin_code] if origin_code in df['ORIGIN'].values else df
                        else:
                            origin_data = df
                        
                        if 'DEST' in df.columns:
                            dest_data = df[df['DEST'] == dest_code] if dest_code in df['DEST'].values else df
                        else:
                            dest_data = df
                    else:
                        origin_data = df
                        dest_data = df
                    
                    # Add all numeric features with default/median values
                    for feature in numeric_features:
                        if feature == 'temp':
                            input_data[feature] = temp
                        elif feature == 'wdsp':
                            input_data[feature] = wdsp
                        elif feature == 'visib' and visib is not None:
                            input_data[feature] = visib
                        elif feature == 'dewp' and dewp is not None:
                            input_data[feature] = dewp
                        elif feature == 'DEP_HOUR':
                            input_data[feature] = dep_hour
                        elif feature in ['MONTH', 'QUARTER', 'YEAR']:
                            # Use current date for time-based features (not user input)
                            from datetime import datetime
                            now = datetime.now()
                            if feature == 'MONTH':
                                input_data[feature] = now.month
                            elif feature == 'QUARTER':
                                input_data[feature] = (now.month - 1) // 3 + 1
                            elif feature == 'YEAR':
                                input_data[feature] = now.year
                        elif feature == 'fog':
                            input_data[feature] = 1 if fog else 0
                        elif feature == 'rain_drizzle':
                            input_data[feature] = 1 if rain else 0
                        elif feature == 'snow_ice_pellets':
                            input_data[feature] = 1 if snow else 0
                        elif feature == 'hail':
                            input_data[feature] = 1 if hail else 0
                        elif feature == 'thunder':
                            input_data[feature] = 1 if thunder else 0
                        elif feature == 'tornado_funnel_cloud':
                            input_data[feature] = 1 if tornado else 0
                        elif feature == 'ORIGIN' or feature.startswith('ORIGIN_'):
                            # If ORIGIN is numeric (encoded), use median from origin airport data
                            if feature in origin_data.columns and origin_data[feature].dtype in [np.number]:
                                input_data[feature] = origin_data[feature].median() if len(origin_data) > 0 else df[feature].median()
                            else:
                                input_data[feature] = df[feature].median() if feature in df.columns else 0
                        elif feature == 'DEST' or feature.startswith('DEST_'):
                            # If DEST is numeric (encoded), use median from dest airport data
                            if feature in dest_data.columns and dest_data[feature].dtype in [np.number]:
                                input_data[feature] = dest_data[feature].median() if len(dest_data) > 0 else df[feature].median()
                            else:
                                input_data[feature] = df[feature].median() if feature in df.columns else 0
                        else:
                            # Use median for other features, prefer origin airport data if available
                            if feature in origin_data.columns and origin_data[feature].dtype in [np.number]:
                                input_data[feature] = origin_data[feature].median() if len(origin_data) > 0 else df[feature].median()
                            elif feature in df.columns:
                                input_data[feature] = df[feature].median()
                            else:
                                input_data[feature] = 0
                    
                    # Create input DataFrame
                    input_df = pd.DataFrame([input_data])
                    input_df = input_df[numeric_features]  # Ensure correct order
                    
                    # Scale using the same scaler (reuse cached scaler)
                    if 'scaler' in st.session_state and st.session_state.scaler is not None:
                        input_scaled = st.session_state.scaler.transform(input_df)
                    else:
                        # Fallback: create new scaler if not available
                        scaler = StandardScaler()
                        X_all = df[numeric_features].fillna(0)
                        scaler.fit(X_all)
                        input_scaled = scaler.transform(input_df)
                        st.session_state.scaler = scaler  # Cache it for next time
                    
                    # Get predictions from all models
                    predictions = {}
                    
                    for model_name, result in st.session_state.results.items():
                        if 'error' not in result and 'model' in result:
                            model = result['model']
                            try:
                                if hasattr(model, 'predict_proba'):
                                    prob = model.predict_proba(input_scaled)[0][1]
                                    predictions[model_name] = prob
                                else:
                                    pred = model.predict(input_scaled)[0]
                                    predictions[model_name] = float(pred)
                            except Exception as e:
                                st.warning(f"Error predicting with {model_name}: {str(e)}")
                    
                    if predictions:
                        # Display results - Minimal design
                        st.markdown("---")
                        st.markdown("### Prediction Results")
                        
                        # Show route if airports were selected
                        if origin_code and dest_code:
                            st.info(f"✈️ Route: {origin_code} → {dest_code}")
                        
                        avg_prob = np.mean(list(predictions.values()))
                        
                        # Main metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Delay Probability", f"{avg_prob:.1%}")
                        with col2:
                            risk_level = "Low" if avg_prob < 0.2 else "Medium" if avg_prob < 0.4 else "High"
                            st.metric("Risk Level", risk_level)
                        with col3:
                            st.metric("Models Used", len(predictions))
                        
                        # Model predictions table
                        pred_df = pd.DataFrame({
                            'Model': list(predictions.keys()),
                            'Probability': [f"{p:.1%}" for p in predictions.values()]
                        })
                        st.dataframe(pred_df, use_container_width=True, hide_index=True)
                        
                        # Chart and Map side by side
                        chart_col, map_col = st.columns(2)
                        
                        with chart_col:
                            # Simple chart
                            fig_pred = px.bar(
                                x=list(predictions.keys()),
                                y=list(predictions.values()),
                                labels={'x': 'Model', 'y': 'Delay Probability'},
                                text=[f"{p:.1%}" for p in predictions.values()]
                            )
                            fig_pred.update_traces(textposition='outside')
                            fig_pred.update_layout(
                                yaxis_tickformat='.0%',
                                showlegend=False,
                                height=400,
                                margin=dict(l=0, r=0, t=20, b=0),
                                title="Delay Probability by Model"
                            )
                            st.plotly_chart(fig_pred, use_container_width=True)
                        
                        with map_col:
                            # Route map
                            if origin_code and dest_code:
                                route_map = create_route_map(origin_code, dest_code, avg_prob, df)
                                if route_map:
                                    st.plotly_chart(route_map, use_container_width=True)
                                else:
                                    st.info(f"📍 Map unavailable for {origin_code} → {dest_code}")
                            else:
                                st.info("📍 Select airports to view route map")
                        
                        st.markdown("---")
                        
                        # Weather impact analysis - Minimal
                        st.markdown("### Weather Impact")
                        
                        baseline_input = input_data.copy()
                        for event in ['fog', 'rain_drizzle', 'snow_ice_pellets', 'hail', 'thunder', 'tornado_funnel_cloud']:
                            if event in baseline_input:
                                baseline_input[event] = 0
                        
                        baseline_df = pd.DataFrame([baseline_input])[numeric_features]
                        if 'scaler' in st.session_state:
                            baseline_scaled = st.session_state.scaler.transform(baseline_df)
                        else:
                            baseline_scaled = scaler.transform(baseline_df)
                        
                        baseline_preds = {}
                        for model_name, result in st.session_state.results.items():
                            if 'error' not in result and 'model' in result:
                                model = result['model']
                                try:
                                    if hasattr(model, 'predict_proba'):
                                        baseline_prob = model.predict_proba(baseline_scaled)[0][1]
                                        baseline_preds[model_name] = baseline_prob
                                except:
                                    pass
                        
                        if baseline_preds:
                            baseline_avg = np.mean(list(baseline_preds.values()))
                            impact = avg_prob - baseline_avg
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Baseline", f"{baseline_avg:.1%}")
                            with col2:
                                st.metric("Current", f"{avg_prob:.1%}")
                            with col3:
                                st.metric("Impact", f"{impact:+.1%}")
                            
                            if impact > 0.1:
                                st.warning(f"Weather increases delay risk by {impact:.1%}")
                            elif impact < -0.05:
                                st.success(f"Weather reduces delay risk by {abs(impact):.1%}")
                            else:
                                st.info("Weather has minimal impact")
                        
                        # Recommendations - Minimal
                        recommendations = []
                        if avg_prob > 0.4:
                            recommendations.append("High delay risk - Consider rescheduling")
                        if thunder:
                            recommendations.append("Thunder detected - Significant delay risk")
                        if snow:
                            recommendations.append("Snow/Ice conditions - High delay probability")
                        if wdsp > 30:
                            recommendations.append(f"High wind speed ({wdsp:.0f} mph) may cause delays")
                        if temp < 20 or temp > 90:
                            recommendations.append(f"Extreme temperature ({temp:.0f}°F) may affect operations")
                        if avg_prob < 0.15:
                            recommendations.append("Low delay risk - Conditions are favorable")
                        
                        if recommendations:
                            st.markdown("### Recommendations")
                            for rec in recommendations:
                                st.write(f"• {rec}")
                        else:
                            st.info("Weather conditions are within normal parameters")


if __name__ == "__main__":
    main()

