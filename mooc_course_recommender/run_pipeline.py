from src.config import OUTPUT_DIR, PROCESSED_DIR, ensure_directories
from src.config import COURSES_FILE, INTERACTIONS_FILE, USERS_FILE
from src.generate_sample_data import main as generate_sample_data
from src.load_data import create_spark_session, load_raw_data
from src.preprocess import preprocess_and_save
from src.train_als import train_from_processed


def main() -> None:
    ensure_directories()
    if not (USERS_FILE.exists() and COURSES_FILE.exists() and INTERACTIONS_FILE.exists()):
        print("Raw data is missing. Generating fallback sample data.")
        generate_sample_data()

    spark = create_spark_session("MoocRecommendationPipeline")
    try:
        users, courses, interactions = load_raw_data(spark)
        tables = preprocess_and_save(users, courses, interactions)
        print("Processed tables saved:")
        for name, df in tables.items():
            print(f"  - {name}: {df.count()} rows")

        _, metrics_df = train_from_processed(spark)
        print("Latest training metrics:")
        print(metrics_df.to_string(index=False))
        print(f"Processed data: {PROCESSED_DIR}")
        print(f"Recommendation output: {OUTPUT_DIR}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
