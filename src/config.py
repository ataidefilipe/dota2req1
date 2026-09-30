import os
import logging
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Project configuration loaded from environment variables."""
    
    OPENDOTA_API_KEY = os.getenv("OPENDOTA_API_KEY", "")
    OPENDOTA_RPS = float(os.getenv("OPENDOTA_RPS", "1.0"))
    
    # Use absolute path if possible or relative to execution dir
    SQLITE_PATH = os.getenv("SQLITE_PATH", "./data/opendota.sqlite")
    
    PUBLICMATCHES_MIN_RANK = int(os.getenv("PUBLICMATCHES_MIN_RANK", "70"))
    PUBLICMATCHES_MAX_RANK = os.getenv("PUBLICMATCHES_MAX_RANK", None) # Optional
    
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    @classmethod
    def validate(cls):
        """Basic validation of configuration."""
        if cls.OPENDOTA_RPS <= 0:
            raise ValueError("OPENDOTA_RPS must be positive.")
        if cls.PUBLICMATCHES_MIN_RANK < 0:
            raise ValueError("PUBLICMATCHES_MIN_RANK must be positive.")
        
        logging.info(f"Config loaded: SQLITE_PATH={cls.SQLITE_PATH}, RPS={cls.OPENDOTA_RPS}")

if __name__ == "__main__":
    # Smoke test for config
    import sys
    logging.basicConfig(level=logging.INFO)
    try:
        Config.validate()
        print("Configuration valid.")
    except Exception as e:
        print(f"Configuration invalid: {e}")
        sys.exit(1)
