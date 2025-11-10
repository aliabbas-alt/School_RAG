# Embedding provider + model defaults
DEFAULT_EMBED_PROVIDER = "openai"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"

# Define a dimensions override here (None = use model default).
# For OpenAI v3 embeddings: must be <= model's default dims
#   - 3-small: 1536
#   - 3-large: 3072
# Example: DEFAULT_EMBED_DIMENSIONS = 512
DEFAULT_EMBED_DIMENSIONS = None

# Output format for embeddings file (".parquet" recommended)
DEFAULT_EMBED_OUTPUT_EXT = ".parquet"

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 180