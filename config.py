import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── McAuley Dataset URLs ───────────────────────────────────────────────────
REVIEWS_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/steam/australian_user_reviews.json.gz"
ITEMS_URL   = "https://mcauleylab.ucsd.edu/public_datasets/data/steam/australian_users_items.json.gz"

REVIEWS_PATH = DATA_DIR / "australian_user_reviews.json.gz"
ITEMS_PATH   = DATA_DIR / "australian_users_items.json.gz"
SURVIVAL_DF_PATH = RESULTS_DIR / "survival_dataset.parquet"
FEATURES_PATH    = RESULTS_DIR / "features_llm.parquet"
MODEL_BASELINE   = RESULTS_DIR / "cox_baseline.pkl"
MODEL_AUGMENTED  = RESULTS_DIR / "cox_augmented.pkl"
EXPERIMENT_PATH  = RESULTS_DIR / "experiment_results.json"

# ── Dataset ────────────────────────────────────────────────────────────────
MAX_REVIEWS    = int(os.getenv("MAX_REVIEWS", 10000))   # cap for LLM cost control
MIN_DURATION   = 0.5                                     # hours, filter out near-zero tenure
MAX_DURATION   = 5000                                    # hours, filter extreme outliers

# ── LLM ───────────────────────────────────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
LLM_MODEL         = "meta-llama/llama-4-scout-17b-16e-instruct"  # 30K TPM free tier (5x vs llama-3.1-8b)
LLM_BATCH_SIZE    = int(os.getenv("LLM_BATCH_SIZE", 10))
LLM_MAX_TOKENS    = 1024

# ── Model ─────────────────────────────────────────────────────────────────
CV_FOLDS          = 5
RANDOM_STATE      = 42
PENALIZER         = 0.1                                  # L2 regularization for Cox PH