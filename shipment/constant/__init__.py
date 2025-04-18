import os
from dotenv import load_dotenv
from datetime import datetime
from from_root.root import from_root



load_dotenv()

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

db_url = os.getenv("MONGO_DB_URL")

DB_URL=db_url

MODEL_CONFIG_FILE = "config/model.yaml"
SCHEMA_FILE_PATH = "config/schema.yaml"
TARGET_COLUMN = "Cost"
DB_NAME = "shipping"
COLLECTION_NAME="shipping_data"
TEST_SIZE=0.2
ARTIFACTS_DIR = os.path.join(from_root(), 'artifacts', TIMESTAMP)

DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"
DATA_INGESTION_TRAIN_DIR = "Train"
DATA_INGESTION_TEST_DIR = "Test"
DATA_INGESTION_TRAIN_FILE_NAME = "train.csv"
DATA_INGESTION_TEST_FILE_NAME = "test.csv"