import os
import yaml
from dotenv import load_dotenv

# Load environment variables (optional override)
load_dotenv()


def load_config(config_path="config.yaml"):
    """
    Loads YAML configuration and merges environment overrides if available.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply environment overrides (if defined)
    db_path_env = os.getenv("DB_PATH")
    if db_path_env:
        config["database"]["folder"] = os.path.dirname(db_path_env)
        config["database"]["name"] = os.path.basename(db_path_env)

    return config
