import os
import sys
import math
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import DEFAULT_MAX_ITER, DEFAULT_RANK, DEFAULT_REG_PARAM, DEFAULT_TOP_N, OUTPUT_DIR, PROCESSED_DIR
from src.load_data import create_spark_session
from src.train_als import train_from_processed


class TrainRequest(BaseModel):
    rank: int = Field(DEFAULT_RANK, ge=2, le=50)
    max_iter: int = Field(DEFAULT_MAX_ITER, ge=3, le=30)
    reg_param: float = Field(DEFAULT_REG_PARAM, ge=0.001, le=1.0)
    top_n: int = Field(DEFAULT_TOP_N, ge=3, le=30)


class AiMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=8000)


class AiChatRequest(BaseModel):
    messages: list[AiMessage] = Field(..., min_length=1, max_length=20)
    context: dict[str, Any] | None = None


def cors_origins() -> list[str]:
    configured = os.getenv("CORS_ALLOW_ORIGINS", "")
    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    return origins or [
        "*",
    ]


app = FastAPI(title="MOOC Course Recommender API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_origin_regex=os.getenv("CORS_ALLOW_ORIGIN_REGEX"),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def table_exists(base_dir: Path, name: str) -> bool:
    return (base_dir / f"{name}.parquet").exists() or (base_dir / f"{name}.csv").exists()


def read_table(base_dir: Path, name: str) -> pd.DataFrame:
    parquet_path = base_dir / f"{name}.parquet"
    csv_path = base_dir / f"{name}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if limit is not None:
        df = df.head(limit)
    rows: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        rows.append({key: clean_value(value) for key, value in row.items()})
    return rows


def first_row(df: pd.DataFrame) -> dict[str, Any]:
    return records(df.head(1))[0] if not df.empty else {}


def require_table(base_dir: Path, name: str) -> pd.DataFrame:
    df = read_table(base_dir, name)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Missing or empty table: {name}")
    return df


def read_deepseek_api_key() -> str:
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key

    for path in [PROJECT_ROOT / "api_key", PROJECT_ROOT.parent / "api_key"]:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()

    raise HTTPException(status_code=500, detail="Missing DeepSeek API key. Put it in api_key or set DEEPSEEK_API_KEY.")


def course_rows_by_ids(course_ids: list[str]) -> list[dict[str, Any]]:
    if not course_ids:
        return []
    courses = read_table(PROJECT_ROOT / "data" / "raw", "courses")
    if courses.empty:
        courses = read_table(PROCESSED_DIR, "courses_clean")
    if courses.empty or "course_id" not in courses.columns:
        return []

    unique_ids = list(dict.fromkeys([course_id for course_id in course_ids if course_id]))
    matched = courses[courses["course_id"].isin(unique_ids)].drop_duplicates("course_id")
    lookup = {row["course_id"]: row for row in records(matched)}
    return [lookup[course_id] for course_id in unique_ids if course_id in lookup]


def format_course_context(context: dict[str, Any] | None) -> str:
    if not context:
        return ""

    selected_ids = context.get("selected_course_ids") or []
    recommended_id = context.get("recommended_course_id")
    selected_courses = course_rows_by_ids(selected_ids)
    recommended_courses = course_rows_by_ids([recommended_id] if recommended_id else [])

    lines = ["## 推荐解释上下文"]
    if recommended_courses:
        course = recommended_courses[0]
        lines.extend(
            [
                f"- 推荐课程：{course.get('course_name') or context.get('recommended_course_name') or recommended_id}",
                f"- 课程ID：{course.get('course_id') or recommended_id}",
                f"- 类别：{course.get('category') or '未分类'}",
                f"- 教师/来源：{course.get('teacher') or '未知'}",
                f"- 课程简介：{course.get('description') or '暂无简介'}",
            ]
        )
    elif recommended_id or context.get("recommended_course_name"):
        lines.append(f"- 推荐课程：{context.get('recommended_course_name') or recommended_id}")

    if context.get("recommendation_score") is not None:
        lines.append(f"- 推荐分数：{context.get('recommendation_score')}")

    if selected_courses:
        lines.append("- 已选择/已学习课程：")
        for course in selected_courses[:12]:
            lines.append(
                "  - "
                f"{course.get('course_name') or course.get('course_id')} "
                f"（{course.get('category') or '未分类'}）："
                f"{course.get('description') or '暂无简介'}"
            )

    return "\n".join(lines)


def call_deepseek(messages: list[dict[str, str]]) -> str:
    api_key = read_deepseek_api_key()
    base_url = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com").strip().rstrip("/")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.65,
        "max_tokens": 1400,
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore") or str(exc)
        raise HTTPException(status_code=502, detail=f"DeepSeek API error: {detail}") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"Cannot reach DeepSeek API: {exc.reason}") from exc

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="DeepSeek API returned an unexpected response") from exc


def histogram(df: pd.DataFrame, column: str, bins: int = 12) -> list[dict[str, Any]]:
    if df.empty or column not in df.columns:
        return []
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return []
    counts = pd.cut(series, bins=bins).value_counts().sort_index()
    return [
        {
            "range": f"{interval.left:.0f}-{interval.right:.0f}",
            "count": int(count),
        }
        for interval, count in counts.items()
    ]


def focused_histogram(df: pd.DataFrame, column: str, bins: int = 12, quantile: float = 0.99) -> dict[str, Any]:
    if df.empty or column not in df.columns:
        return {"rows": [], "tail": None}

    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return {"rows": [], "tail": None}

    lower = max(0, int(math.floor(series.min())))
    upper = int(math.ceil(series.quantile(quantile)))
    upper = max(upper, lower + 1)
    focused = series[series <= upper]
    hidden = series[series > upper]

    edge_count = min(bins, max(1, upper - lower)) + 1
    edges = np.unique(np.linspace(lower, upper, edge_count))
    counts = pd.cut(focused, bins=edges, include_lowest=True).value_counts().sort_index()
    rows = []
    for interval, count in counts.items():
        left = max(lower, int(math.floor(interval.left)))
        right = int(math.ceil(interval.right))
        rows.append(
            {
                "range": f"{left}-{right}",
                "midpoint": round((left + right) / 2, 1),
                "count": int(count),
            }
        )

    tail = None
    if not hidden.empty:
        tail = {
            "range": f"{int(hidden.min())}-{int(hidden.max())}",
            "count": int(hidden.count()),
            "min": int(hidden.min()),
            "max": int(hidden.max()),
        }

    return {"rows": rows, "tail": tail}


def grouped_categories(data: pd.DataFrame, value_col: str) -> list[dict[str, Any]]:
    if data.empty or "category" not in data.columns or value_col not in data.columns:
        return []
    threshold_rows = data[data["category"] == "社会学"]
    if threshold_rows.empty:
        display = data[["category", value_col]].copy()
        display["category_display"] = display["category"]
    else:
        threshold = threshold_rows[value_col].max()
        display = data[["category", value_col]].copy()
        display["category_display"] = display["category"].where(display[value_col] >= threshold, "其他")
    grouped = display.groupby("category_display", as_index=False)[value_col].sum().sort_values(value_col, ascending=False)
    return records(grouped.rename(columns={"category_display": "category"}))


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


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "processed_ready": table_exists(PROCESSED_DIR, "stats_summary"),
        "recommendations_ready": table_exists(OUTPUT_DIR, "recommendations"),
    }


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    summary = require_table(PROCESSED_DIR, "stats_summary")
    popular_courses = require_table(PROCESSED_DIR, "popular_courses")
    category_distribution = require_table(PROCESSED_DIR, "category_distribution")
    user_course_counts = require_table(PROCESSED_DIR, "user_course_counts")
    course_learn_counts = require_table(PROCESSED_DIR, "course_learn_counts")
    return {
        "summary": first_row(summary),
        "popular_courses": records(popular_courses.sort_values("learn_count", ascending=False), 10),
        "category_distribution": grouped_categories(category_distribution, "interaction_count"),
        "user_course_histogram": focused_histogram(user_course_counts, "course_count", bins=12, quantile=0.99),
        "course_learn_histogram": focused_histogram(course_learn_counts, "learn_count", bins=12, quantile=0.99),
    }


@app.get("/api/training/latest")
def latest_training() -> dict[str, Any]:
    latest_metrics = read_table(OUTPUT_DIR, "latest_metrics")
    history = read_table(OUTPUT_DIR, "metrics_history")
    return {
        "latest": first_row(latest_metrics),
        "history": records(history.sort_values("trained_at", ascending=False), 20) if not history.empty else [],
    }


@app.post("/api/training/train")
def train_model(request: TrainRequest) -> dict[str, Any]:
    spark = create_spark_session("MoocCourseRecommenderApi")
    _, metrics_df = train_from_processed(
        spark,
        request.rank,
        request.max_iter,
        request.reg_param,
        request.top_n,
    )
    return {"metrics": records(metrics_df)}


@app.get("/api/recommendations/users")
def recommendation_users(limit: int = Query(200, ge=1, le=1000)) -> dict[str, Any]:
    recommendations = require_table(OUTPUT_DIR, "recommendations")
    users = sorted(recommendations["user_id"].dropna().unique().tolist())[:limit]
    return {"users": users}


@app.get("/api/recommendations/by-user")
def recommendations_by_user(
    user_id: str = Query(..., min_length=1),
    top_n: int = Query(DEFAULT_TOP_N, ge=3, le=30),
) -> dict[str, Any]:
    recommendations = require_table(OUTPUT_DIR, "recommendations")
    user_recs = (
        recommendations[recommendations["user_id"] == user_id]
        .sort_values("recommendation_score", ascending=False)
        .head(top_n)
    )
    if user_recs.empty:
        raise HTTPException(status_code=404, detail=f"No recommendations for user_id: {user_id}")
    return {"items": records(user_recs)}


@app.get("/api/courses")
def courses(limit: int = Query(500, ge=1, le=5000)) -> dict[str, Any]:
    interactions = require_table(PROCESSED_DIR, "interactions_clean")
    course_dim = require_table(PROCESSED_DIR, "courses_clean")
    learned_courses = (
        interactions[["course_id"]]
        .drop_duplicates()
        .merge(course_dim[["course_id", "course_name", "category"]], on="course_id", how="left")
        .sort_values(["category", "course_name"])
        .head(limit)
    )
    return {"courses": records(learned_courses)}


@app.get("/api/recommendations/by-courses")
def recommendations_by_courses(
    course_ids: list[str] = Query(..., min_length=1),
    top_n: int = Query(DEFAULT_TOP_N, ge=3, le=30),
) -> dict[str, Any]:
    interactions = require_table(PROCESSED_DIR, "interactions_clean")
    course_dim = require_table(PROCESSED_DIR, "courses_clean")
    recs = recommend_by_selected_courses(course_ids, interactions, course_dim, top_n)
    if recs.empty:
        raise HTTPException(status_code=404, detail="No recommendations for selected courses")
    return {"items": records(recs)}


@app.post("/api/ai/chat")
def ai_chat(request: AiChatRequest) -> dict[str, Any]:
    context_text = format_course_context(request.context)
    system_prompt = (
        "你是 MOOC 课程推荐系统中的 AI 学业顾问。"
        "回答必须使用中文 Markdown，结构清晰、语气积极、有说服力。"
        "解释课程时要优先依据课程 description，不要编造没有依据的事实。"
        "如果存在推荐上下文，请结合已选择课程、推荐课程和推荐分数，说明这门课讲什么、为什么适合继续学习、能带来什么学习收益。"
        "不要说自己无法访问外部系统；只基于用户提供和系统上下文回答。"
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if context_text:
        messages.append({"role": "system", "content": context_text})
    messages.extend([message.model_dump() for message in request.messages])
    answer = call_deepseek(messages)
    return {"content": answer}


@app.get("/api/analysis")
def analysis() -> dict[str, Any]:
    history = read_table(OUTPUT_DIR, "metrics_history")
    recommendations = read_table(OUTPUT_DIR, "recommendations")
    popular_courses = read_table(PROCESSED_DIR, "popular_courses")
    courses_df = read_table(PROCESSED_DIR, "courses_clean")

    latest = first_row(history.sort_values("trained_at", ascending=False)) if not history.empty else {}
    total_course_count = int(courses_df["course_id"].nunique()) if not courses_df.empty else 0
    recommended_course_count = int(recommendations["course_id"].nunique()) if not recommendations.empty else 0
    avg_recommend_count = round(len(recommendations) / recommended_course_count, 1) if recommended_course_count else 0
    hot_recs = pd.DataFrame()
    if not recommendations.empty:
        hot_recs = (
            recommendations.groupby(["course_id", "course_name", "category"], as_index=False)
            .size()
            .sort_values("size", ascending=False)
            .head(10)
        )

    return {
        "latest": latest,
        "summary": {
            "total_course_count": total_course_count,
            "recommended_course_count": recommended_course_count,
            "avg_recommend_count": avg_recommend_count,
        },
        "history": records(history.sort_values("trained_at", ascending=False), 20) if not history.empty else [],
        "hot_recommendations": records(hot_recs),
        "popular_courses": records(popular_courses.sort_values("learn_count", ascending=False), 10) if not popular_courses.empty else [],
    }
