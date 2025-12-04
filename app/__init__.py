from __future__ import annotations

# Load environment variables as early as possible so all app submodules see
# .env overrides even when they read defaults at import time.
from dotenv import load_dotenv

load_dotenv(override=False)
