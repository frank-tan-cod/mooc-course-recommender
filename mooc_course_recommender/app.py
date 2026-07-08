import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))

from src.config import DEFAULT_MAX_ITER, DEFAULT_RANK, DEFAULT_REG_PARAM, DEFAULT_TOP_N, OUTPUT_DIR, PROCESSED_DIR
from src.generate_sample_data import main as generate_sample_data
from src.load_data import create_spark_session, load_raw_data
from src.preprocess import preprocess_and_save
from src.train_als import train_from_processed


st.set_page_config(page_title="慕课课程协同过滤推荐系统", layout="wide")


@st.cache_resource
def get_spark():
    return create_spark_session("MoocCourseRecommenderStreamlit")


def table_exists(base_dir: Path, name: str) -> bool:
    return (base_dir / f"{name}.parquet").exists() or (base_dir / f"{name}.csv").exists()


@st.cache_data(show_spinner=False)
def read_table(base_dir: str, name: str) -> pd.DataFrame:
    base = Path(base_dir)
    parquet_path = base / f"{name}.parquet"
    csv_path = base / f"{name}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def invalidate_cache() -> None:
    read_table.clear()


def initialize_processed_data() -> None:
    generate_sample_data()
    spark = get_spark()
    users, courses, interactions = load_raw_data(spark)
    preprocess_and_save(users, courses, interactions)
    invalidate_cache()


def train_model(rank: int, max_iter: int, reg_param: float, top_n: int) -> pd.DataFrame:
    spark = get_spark()
    _, metrics_df = train_from_processed(spark, rank, max_iter, reg_param, top_n)
    invalidate_cache()
    return metrics_df


def recommend_by_selected_courses(
    selected_course_ids: list[str],
    interactions: pd.DataFrame,
    courses: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    if not selected_course_ids or interactions.empty or courses.empty:
        return pd.DataFrame()

    selected = set(selected_course_ids)
    matched = interactions[interactions["course_id"].isin(selected)]
    if matched.empty:
        return pd.DataFrame()

    user_weights = (
        matched.groupby("user_id")
        .agg(matched_course_count=("course_id", "nunique"), matched_preference=("preference", "sum"))
        .reset_index()
    )
    user_weights["user_weight"] = user_weights["matched_course_count"] + user_weights["matched_preference"] * 0.1

    candidates = interactions.merge(user_weights[["user_id", "user_weight"]], on="user_id", how="inner")
    candidates = candidates[~candidates["course_id"].isin(selected)]
    if candidates.empty:
        return pd.DataFrame()

    scored = (
        candidates.groupby("course_id")
        .agg(
            similar_user_count=("user_id", "nunique"),
            cooccurrence_score=("user_weight", "sum"),
            avg_preference=("preference", "mean"),
        )
        .reset_index()
    )
    scored["recommendation_score"] = (
        scored["cooccurrence_score"] * 0.7
        + scored["similar_user_count"] * 0.2
        + scored["avg_preference"] * 0.1
    )

    course_dim = courses[["course_id", "course_name", "category", "teacher"]].drop_duplicates("course_id")
    result = scored.merge(course_dim, on="course_id", how="left")
    result = result.sort_values(
        ["recommendation_score", "similar_user_count", "avg_preference"],
        ascending=False,
    ).head(top_n)
    result.insert(0, "rank_position", range(1, len(result) + 1))
    return result[
        [
            "rank_position",
            "course_id",
            "course_name",
            "category",
            "teacher",
            "similar_user_count",
            "avg_preference",
            "recommendation_score",
        ]
    ]


def require_processed_data() -> bool:
    required = ["stats_summary", "popular_courses", "category_distribution", "user_course_counts", "als_train_data"]
    missing = [name for name in required if not table_exists(PROCESSED_DIR, name)]
    if missing:
        st.warning("尚未检测到 Spark 预处理结果，请先生成示例数据并执行预处理。")
        if st.button("生成示例数据并预处理", type="primary"):
            with st.spinner("正在调用 Spark 完成数据接入与预处理..."):
                initialize_processed_data()
            st.success("数据已生成并预处理完成。")
            st.rerun()
        return False
    return True


def metric_card(label: str, value) -> None:
    st.metric(label, value if value is not None else "-")


def overview_page() -> None:
    st.subheader("数据概览")
    if not require_processed_data():
        return

    summary = read_table(str(PROCESSED_DIR), "stats_summary")
    popular_courses = read_table(str(PROCESSED_DIR), "popular_courses")
    category_distribution = read_table(str(PROCESSED_DIR), "category_distribution")
    user_course_counts = read_table(str(PROCESSED_DIR), "user_course_counts")
    course_learn_counts = read_table(str(PROCESSED_DIR), "course_learn_counts")

    if not summary.empty:
        row = summary.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("用户数量", int(row["user_count"]))
        col2.metric("课程数量", int(row["course_count"]))
        col3.metric("交互记录", int(row["interaction_count"]))
        col4.metric("平均偏好", row.get("avg_preference", "-"))

    col1, col2 = st.columns(2)
    with col1:
        top10 = popular_courses.head(10).sort_values("learn_count", ascending=True)
        fig = px.bar(
            top10,
            x="learn_count",
            y="course_name",
            orientation="h",
            color="category",
            title="热门课程 Top 10",
            labels={"learn_count": "学习次数", "course_name": "课程"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.pie(
            category_distribution,
            names="category",
            values="interaction_count",
            title="课程类别交互分布",
            hole=0.35,
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.histogram(
            user_course_counts,
            x="course_count",
            nbins=20,
            title="用户选课数量分布",
            labels={"course_count": "选课数量"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.histogram(
            course_learn_counts,
            x="learn_count",
            nbins=20,
            title="课程被学习次数分布",
            labels={"learn_count": "学习次数"},
        )
        st.plotly_chart(fig, use_container_width=True)


def training_page() -> None:
    st.subheader("模型训练")
    if not require_processed_data():
        return

    col1, col2, col3, col4 = st.columns(4)
    rank = col1.number_input("rank", min_value=2, max_value=50, value=DEFAULT_RANK, step=1)
    max_iter = col2.number_input("maxIter", min_value=3, max_value=30, value=DEFAULT_MAX_ITER, step=1)
    reg_param = col3.number_input("regParam", min_value=0.001, max_value=1.0, value=DEFAULT_REG_PARAM, step=0.01, format="%.3f")
    top_n = col4.number_input("topN", min_value=3, max_value=30, value=DEFAULT_TOP_N, step=1)

    if st.button("训练 ALS 推荐模型", type="primary"):
        with st.spinner("正在调用 Spark MLlib ALS 训练模型并写入推荐结果..."):
            metrics = train_model(int(rank), int(max_iter), float(reg_param), int(top_n))
        st.success("训练完成，推荐结果和指标已保存到 data/output。")
        st.dataframe(metrics, use_container_width=True)

    latest_metrics = read_table(str(OUTPUT_DIR), "latest_metrics")
    if not latest_metrics.empty:
        st.markdown("#### 最近一次训练指标")
        cols = st.columns(4)
        latest = latest_metrics.iloc[0]
        cols[0].metric("RMSE", latest.get("rmse", "-"))
        cols[1].metric("覆盖率", latest.get("coverage", "-"))
        cols[2].metric("多样性", latest.get("diversity", "-"))
        cols[3].metric("热门推荐占比", latest.get("popular_recommendation_ratio", "-"))
        st.dataframe(latest_metrics, use_container_width=True)


def recommendation_page() -> None:
    st.subheader("个性化推荐")
    if not require_processed_data():
        return

    mode = st.radio("推荐方式", ["按学习者 user_id 推荐", "按已学课程推荐"], horizontal=True)

    if mode == "按学习者 user_id 推荐":
        if not table_exists(OUTPUT_DIR, "recommendations"):
            st.info("请先在模型训练页面训练 ALS 模型生成推荐结果。")
            return

        recommendations = read_table(str(OUTPUT_DIR), "recommendations")
        users = sorted(recommendations["user_id"].dropna().unique().tolist())
        if not users:
            st.warning("推荐结果为空，请重新训练模型。")
            return

        col1, col2 = st.columns([2, 1])
        user_id = col1.selectbox("选择学习者 user_id", users)
        top_n = col2.slider("展示 Top-N", min_value=3, max_value=30, value=min(DEFAULT_TOP_N, 30), key="user_top_n")

        user_recs = recommendations[recommendations["user_id"] == user_id].sort_values("recommendation_score", ascending=False).head(top_n)
        st.dataframe(
            user_recs[["rank_position", "course_id", "course_name", "category", "teacher", "recommendation_score"]],
            use_container_width=True,
            hide_index=True,
        )

        col3, col4 = st.columns(2)
        with col3:
            fig = px.bar(
                user_recs.sort_values("recommendation_score", ascending=True),
                x="recommendation_score",
                y="course_name",
                color="category",
                orientation="h",
                title=f"{user_id} 推荐分数",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            category_counts = user_recs.groupby("category", as_index=False).size()
            fig = px.pie(category_counts, names="category", values="size", title="推荐课程类别分布", hole=0.35)
            st.plotly_chart(fig, use_container_width=True)
        return

    interactions = read_table(str(PROCESSED_DIR), "interactions_clean")
    courses = read_table(str(PROCESSED_DIR), "courses_clean")
    learned_courses = (
        interactions[["course_id"]]
        .drop_duplicates()
        .merge(courses[["course_id", "course_name", "category"]], on="course_id", how="left")
        .sort_values(["category", "course_name"])
    )
    learned_courses["label"] = (
        learned_courses["course_name"].fillna(learned_courses["course_id"])
        + " ["
        + learned_courses["course_id"]
        + "]"
    )

    col1, col2 = st.columns([3, 1])
    selected_labels = col1.multiselect(
        "选择你之前学过的课程",
        learned_courses["label"].tolist(),
        default=learned_courses["label"].head(3).tolist(),
    )
    top_n = col2.slider("展示 Top-N", min_value=3, max_value=30, value=min(DEFAULT_TOP_N, 30), key="course_top_n")
    label_to_id = dict(zip(learned_courses["label"], learned_courses["course_id"]))
    selected_course_ids = [label_to_id[label] for label in selected_labels]

    course_recs = recommend_by_selected_courses(selected_course_ids, interactions, courses, top_n)
    if course_recs.empty:
        st.warning("当前选择课程没有找到足够的共学用户，请增加或更换已学课程。")
        return

    st.dataframe(course_recs, use_container_width=True, hide_index=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.bar(
            course_recs.sort_values("recommendation_score", ascending=True),
            x="recommendation_score",
            y="course_name",
            color="category",
            orientation="h",
            title="基于已学课程的协同过滤推荐分数",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        category_counts = course_recs.groupby("category", as_index=False).size()
        fig = px.pie(category_counts, names="category", values="size", title="推荐课程类别分布", hole=0.35)
        st.plotly_chart(fig, use_container_width=True)


def analysis_page() -> None:
    st.subheader("结果分析")
    if not require_processed_data():
        return

    history = read_table(str(OUTPUT_DIR), "metrics_history")
    recommendations = read_table(str(OUTPUT_DIR), "recommendations")
    popular_courses = read_table(str(PROCESSED_DIR), "popular_courses")

    if not history.empty:
        st.markdown("#### 参数组合指标对比")
        st.dataframe(history, use_container_width=True)
        fig = px.scatter(
            history,
            x="rank",
            y="rmse",
            color="regParam",
            size="coverage",
            hover_data=["maxIter", "topN", "diversity", "trained_at"],
            title="ALS 参数与 RMSE/覆盖率对比",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无模型指标历史，请先训练模型。")

    if not recommendations.empty:
        col1, col2 = st.columns(2)
        with col1:
            hot_recs = (
                recommendations.groupby(["course_id", "course_name", "category"], as_index=False)
                .size()
                .sort_values("size", ascending=False)
                .head(10)
            )
            fig = px.bar(hot_recs, x="size", y="course_name", color="category", orientation="h", title="推荐结果中出现最多的课程")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            top_hot = popular_courses.head(10)[["course_name", "learn_count", "category"]]
            fig = px.bar(top_hot, x="learn_count", y="course_name", color="category", orientation="h", title="原始交互热门课程")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 应用案例")
    st.write(
        "在线教育平台可将学习者历史选课、观看视频和完成练习等行为接入 Spark，"
        "清洗后构造用户-课程隐式偏好矩阵。ALS 模型会学习用户与课程的潜在兴趣因子，"
        "当学习者进入平台时，系统根据其历史行为生成个性化 Top-N 慕课推荐，帮助其继续学习相关课程。"
    )


def main() -> None:
    st.title("基于 Spark 的慕课课程协同过滤推荐系统")
    page = st.sidebar.radio("功能导航", ["数据概览", "模型训练", "个性化推荐", "结果分析"])
    st.sidebar.caption("数据来自 Spark 处理后的 processed/output 表")

    if page == "数据概览":
        overview_page()
    elif page == "模型训练":
        training_page()
    elif page == "个性化推荐":
        recommendation_page()
    else:
        analysis_page()


if __name__ == "__main__":
    main()
