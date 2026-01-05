"""
Configuration file for API keys and other settings.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API Configuration (deprecated, use ZAI_API_KEY instead)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Z.ai API Configuration
ZAI_API_KEY = os.environ.get("ZAI_API_KEY")
ZAI_API_URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"
ZAI_MODEL = "glm-4.7"  # Z.ai 기본 모델

# API Rate Limiting Configuration
CALLS_PER_MINUTE = 20
PERIOD = 60
MAX_RETRIES = 3
RETRY_DELAY = 120

# Model Configuration
DEFAULT_MODEL = "glm-4.7"  # Z.ai 기본 모델
DEFAULT_MAX_TOKENS = 2048

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # please specify your preferred embedding model
EMBEDDING_BATCH_SIZE = 32

# Cache Configuration
CACHE_DIR = "cache"
RESULT_DIR = "result" 