from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- PROJECT DIRECTORIES ---
PROJECT_NAME = "Anomaly-Detection-in-LLM-Operations"
PROJECT_DIR = BASE_DIR

# --- DATA DIRECTORIES ---
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# --- NOTEBOOK DIRECTORIES ---
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# --- MODEL DIRECTORIES ---
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

# --- OUTPUT DIRECTORIES ---
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

# --- LOGGING DIRECTORIES ---
LOGS_DIR = BASE_DIR / "logs"
DATA_PIPELINE_LOG = LOGS_DIR / "data_pipeline.log"
MODEL_PIPELINE_LOG = LOGS_DIR / "model_pipeline.log"
MAIN_PIPELINE_LOG = LOGS_DIR / "main_pipeline.log"
