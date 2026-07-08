from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pyspark.sql import SparkSession

from src.config import (
    COURSES_FILE,
    INTERACTIONS_FILE,
    PROJECT_ROOT,
    USERS_FILE,
    ensure_directories,
    spark_local_config,
)


def create_spark_session(app_name: str = "MoocCourseRecommender") -> SparkSession:
    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.local.hostname", "localhost")
        .config("spark.driver.extraJavaOptions", "-Dfile.encoding=UTF-8")
        .config("spark.executor.extraJavaOptions", "-Dfile.encoding=UTF-8")
    )
    for key, value in spark_local_config().items():
        builder = builder.config(key, value)
    return builder.getOrCreate()


def _read_csv_or_json(spark: SparkSession, path_without_suffix: Path):
    csv_path = path_without_suffix.with_suffix(".csv")
    json_path = path_without_suffix.with_suffix(".json")
    if csv_path.exists():
        return spark.read.option("header", True).option("inferSchema", True).csv(str(csv_path))
    if json_path.exists():
        return spark.read.option("multiLine", True).json(str(json_path))
    raise FileNotFoundError(f"Missing data file: {csv_path.name} or {json_path.name}")


def load_raw_data(spark: SparkSession):
    ensure_directories()
    users = _read_csv_or_json(spark, USERS_FILE.with_suffix(""))
    courses = _read_csv_or_json(spark, COURSES_FILE.with_suffix(""))
    interactions = _read_csv_or_json(spark, INTERACTIONS_FILE.with_suffix(""))
    return users, courses, interactions


if __name__ == "__main__":
    spark = create_spark_session("LoadMoocData")
    users_df, courses_df, interactions_df = load_raw_data(spark)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"users={users_df.count()}, courses={courses_df.count()}, interactions={interactions_df.count()}")
    spark.stop()
