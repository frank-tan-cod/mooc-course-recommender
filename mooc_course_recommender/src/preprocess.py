from pyspark.ml.feature import StringIndexer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import MIN_COURSE_INTERACTIONS, MIN_USER_INTERACTIONS, PROCESSED_DIR
from src.storage import save_table


def clean_interactions(
    interactions: DataFrame,
    min_user_interactions: int = MIN_USER_INTERACTIONS,
    min_course_interactions: int = MIN_COURSE_INTERACTIONS,
) -> DataFrame:
    base = (
        interactions.select(
            F.col("user_id").cast("string").alias("user_id"),
            F.col("course_id").cast("string").alias("course_id"),
            F.coalesce(F.col("preference").cast("double"), F.lit(1.0)).alias("preference"),
        )
        .where(F.col("user_id").isNotNull() & F.col("course_id").isNotNull())
        .groupBy("user_id", "course_id")
        .agg(F.max("preference").alias("preference"))
    )

    active_users = (
        base.groupBy("user_id")
        .count()
        .where(F.col("count") >= min_user_interactions)
        .select("user_id")
    )
    active_courses = (
        base.groupBy("course_id")
        .count()
        .where(F.col("count") >= min_course_interactions)
        .select("course_id")
    )

    return base.join(active_users, "user_id", "inner").join(active_courses, "course_id", "inner")


def build_indexed_tables(interactions_clean: DataFrame):
    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_index", handleInvalid="skip")
    course_indexer = StringIndexer(inputCol="course_id", outputCol="course_index", handleInvalid="skip")

    user_model = user_indexer.fit(interactions_clean)
    with_user = user_model.transform(interactions_clean)
    course_model = course_indexer.fit(with_user)
    indexed = course_model.transform(with_user)

    als_train_data = indexed.select(
        "user_id",
        "course_id",
        F.col("user_index").cast("int").alias("user_index"),
        F.col("course_index").cast("int").alias("course_index"),
        F.col("preference").cast("float").alias("preference"),
    )

    user_index_map = als_train_data.select("user_id", "user_index").distinct()
    course_index_map = als_train_data.select("course_id", "course_index").distinct()
    return user_index_map, course_index_map, als_train_data


def build_statistics(interactions_clean: DataFrame, courses: DataFrame):
    course_dim = courses.select(
        F.col("course_id").cast("string").alias("course_id"),
        F.col("course_name").cast("string").alias("course_name"),
        F.col("category").cast("string").alias("category"),
        F.col("teacher").cast("string").alias("teacher"),
        F.col("description").cast("string").alias("description"),
    ).dropDuplicates(["course_id"])

    summary = interactions_clean.agg(
        F.countDistinct("user_id").alias("user_count"),
        F.countDistinct("course_id").alias("course_count"),
        F.count("*").alias("interaction_count"),
        F.round(F.avg("preference"), 3).alias("avg_preference"),
    )

    popular_courses = (
        interactions_clean.groupBy("course_id")
        .agg(F.count("*").alias("learn_count"), F.round(F.avg("preference"), 3).alias("avg_preference"))
        .join(course_dim, "course_id", "left")
        .orderBy(F.desc("learn_count"), F.desc("avg_preference"))
    )

    category_distribution = (
        interactions_clean.join(course_dim.select("course_id", "category"), "course_id", "left")
        .groupBy("category")
        .agg(F.countDistinct("course_id").alias("course_count"), F.count("*").alias("interaction_count"))
        .orderBy(F.desc("interaction_count"))
    )

    user_course_counts = (
        interactions_clean.groupBy("user_id").agg(F.count("*").alias("course_count")).orderBy("course_count")
    )
    course_learn_counts = (
        interactions_clean.groupBy("course_id")
        .agg(F.count("*").alias("learn_count"))
        .join(course_dim.select("course_id", "course_name", "category"), "course_id", "left")
        .orderBy("learn_count")
    )

    return {
        "stats_summary": summary,
        "popular_courses": popular_courses,
        "category_distribution": category_distribution,
        "user_course_counts": user_course_counts,
        "course_learn_counts": course_learn_counts,
        "courses_clean": course_dim,
    }


def preprocess_and_save(users: DataFrame, courses: DataFrame, interactions: DataFrame) -> dict[str, DataFrame]:
    interactions_clean = clean_interactions(interactions)
    user_index_map, course_index_map, als_train_data = build_indexed_tables(interactions_clean)
    statistics = build_statistics(interactions_clean, courses)

    outputs = {
        "interactions_clean": interactions_clean,
        "user_index_map": user_index_map,
        "course_index_map": course_index_map,
        "als_train_data": als_train_data,
        **statistics,
    }
    for name, df in outputs.items():
        save_table(df, PROCESSED_DIR, name)
    return outputs
