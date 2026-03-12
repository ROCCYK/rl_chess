from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "td_agent.json"

