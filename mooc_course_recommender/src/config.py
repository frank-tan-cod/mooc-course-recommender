from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

USERS_FILE = RAW_DIR / "users.csv"
COURSES_FILE = RAW_DIR / "courses.csv"
INTERACTIONS_FILE = RAW_DIR / "interactions.csv"

MIN_USER_INTERACTIONS = 2
MIN_COURSE_INTERACTIONS = 2
DEFAULT_RANK = 10
DEFAULT_MAX_ITER = 8
DEFAULT_REG_PARAM = 0.08
DEFAULT_TOP_N = 8
DEFAULT_SEED = 42


def ensure_directories() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def spark_local_config() -> dict[str, str]:
    return {
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.sql.shuffle.partitions": "8",
        "spark.ui.showConsoleProgress": "false",
    }
