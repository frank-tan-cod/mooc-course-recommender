from datetime import datetime
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame

from src.config import DEFAULT_SEED, OUTPUT_DIR, PROCESSED_DIR
from src.evaluate import compute_recommendation_metrics, compute_rmse
from src.recommend import enrich_recommendations, explode_recommendations
from src.storage import read_table, save_pandas_table, save_table


def train_als_model(
    als_train_data: DataFrame,
    rank: int,
    max_iter: int,
    reg_param: float,
    seed: int = DEFAULT_SEED,
):
    train_df, test_df = als_train_data.randomSplit([0.82, 0.18], seed=seed)
    als = ALS(
        userCol="user_index",
        itemCol="course_index",
        ratingCol="preference",
        rank=rank,
        maxIter=max_iter,
        regParam=reg_param,
        implicitPrefs=True,
        coldStartStrategy="drop",
        nonnegative=True,
        seed=seed,
    )
    model = als.fit(train_df)
    rmse = compute_rmse(model, test_df)
    return model, train_df, test_df, rmse


def generate_all_recommendations(
    model,
    user_index_map: DataFrame,
    course_index_map: DataFrame,
    courses: DataFrame,
    top_n: int,
) -> DataFrame:
    raw_recs = model.recommendForAllUsers(top_n)
    exploded = explode_recommendations(raw_recs, user_index_map, course_index_map)
    return enrich_recommendations(exploded, courses)


def train_from_processed(
    spark,
    rank: int = 10,
    max_iter: int = 8,
    reg_param: float = 0.08,
    top_n: int = 8,
):
    als_train_data = read_table(spark, PROCESSED_DIR, "als_train_data")
    user_index_map = read_table(spark, PROCESSED_DIR, "user_index_map")
    course_index_map = read_table(spark, PROCESSED_DIR, "course_index_map")
    courses = read_table(spark, PROCESSED_DIR, "courses_clean")

    model, _, _, rmse = train_als_model(als_train_data, rank, max_iter, reg_param)
    recommendations = generate_all_recommendations(model, user_index_map, course_index_map, courses, top_n)

    total_course_count = courses.select("course_id").distinct().count()
    rec_metrics = compute_recommendation_metrics(recommendations, courses, total_course_count)
    metrics = {
        "rmse": round(float(rmse), 4) if rmse is not None else None,
        **rec_metrics,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    params = {
        "rank": int(rank),
        "maxIter": int(max_iter),
        "regParam": float(reg_param),
        "topN": int(top_n),
    }

    save_table(recommendations, OUTPUT_DIR, "recommendations")
    metrics_df = pd.DataFrame([{**params, **metrics}])
    save_pandas_table(metrics_df, OUTPUT_DIR, "latest_metrics")

    history_path = OUTPUT_DIR / "metrics_history.parquet"
    if history_path.exists():
        history = pd.concat([pd.read_parquet(history_path), metrics_df], ignore_index=True)
    else:
        history = metrics_df
    history = history.sort_values("trained_at", ascending=False)
    save_pandas_table(history, OUTPUT_DIR, "metrics_history")

    return recommendations, metrics_df


if __name__ == "__main__":
    from src.load_data import create_spark_session

    spark_session = create_spark_session("TrainALS")
    _, latest_metrics = train_from_processed(spark_session)
    print(latest_metrics.to_string(index=False))
    print(f"Recommendation results saved to {OUTPUT_DIR}")
    spark_session.stop()
