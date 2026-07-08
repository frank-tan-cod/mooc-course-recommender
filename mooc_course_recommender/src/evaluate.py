from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def compute_rmse(model, test_df: DataFrame) -> float | None:
    predictions = model.transform(test_df).na.drop(subset=["prediction"])
    if predictions.limit(1).count() == 0:
        return None
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="preference",
        predictionCol="prediction",
    )
    return float(evaluator.evaluate(predictions))


def compute_recommendation_metrics(recommendations: DataFrame, courses: DataFrame, total_course_count: int) -> dict:
    if recommendations.limit(1).count() == 0:
        return {
            "coverage": 0.0,
            "diversity": 0.0,
            "popular_recommendation_ratio": 0.0,
        }

    recommended_course_count = recommendations.select("course_id").distinct().count()
    coverage = recommended_course_count / total_course_count if total_course_count else 0.0

    with_category = recommendations.select("user_id", "course_id", "category")
    user_category_counts = with_category.groupBy("user_id").agg(F.countDistinct("category").alias("category_count"))
    diversity = user_category_counts.agg(F.avg("category_count")).first()[0] or 0.0

    course_freq = recommendations.groupBy("course_id").agg(F.count("*").alias("recommend_count"))
    threshold = course_freq.approxQuantile("recommend_count", [0.8], 0.05)[0]
    popular_count = course_freq.where(F.col("recommend_count") >= threshold).agg(F.sum("recommend_count")).first()[0] or 0
    total_count = recommendations.count()

    return {
        "coverage": round(float(coverage), 4),
        "diversity": round(float(diversity), 4),
        "popular_recommendation_ratio": round(float(popular_count / total_count), 4) if total_count else 0.0,
    }


def metrics_to_dataframe(spark, metrics: dict, params: dict):
    row = {**params, **metrics}
    return spark.createDataFrame([row])
