from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def explode_recommendations(recommendations: DataFrame, user_index_map: DataFrame, course_index_map: DataFrame) -> DataFrame:
    return (
        recommendations.select("user_index", F.posexplode("recommendations").alias("rank_position", "rec"))
        .select(
            "user_index",
            (F.col("rank_position") + F.lit(1)).alias("rank_position"),
            F.col("rec.course_index").cast("int").alias("course_index"),
            F.col("rec.rating").alias("recommendation_score"),
        )
        .join(user_index_map, "user_index", "left")
        .join(course_index_map, "course_index", "left")
    )


def enrich_recommendations(recommendations: DataFrame, courses: DataFrame) -> DataFrame:
    course_dim = courses.select("course_id", "course_name", "category", "teacher").dropDuplicates(["course_id"])
    return (
        recommendations.join(course_dim, "course_id", "left")
        .select(
            "user_id",
            "course_id",
            "course_name",
            "category",
            "teacher",
            "rank_position",
            F.round("recommendation_score", 6).alias("recommendation_score"),
        )
        .orderBy("user_id", "rank_position")
    )


def recommend_for_user(all_recommendations: DataFrame, user_id: str, top_n: int) -> DataFrame:
    return (
        all_recommendations.where(F.col("user_id") == user_id)
        .orderBy(F.desc("recommendation_score"))
        .limit(top_n)
    )
