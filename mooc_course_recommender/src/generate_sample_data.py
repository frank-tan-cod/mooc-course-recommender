import random
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import COURSES_FILE, INTERACTIONS_FILE, RAW_DIR, USERS_FILE, ensure_directories


CATEGORIES = [
    "人工智能",
    "大数据",
    "软件工程",
    "计算机基础",
    "产品设计",
    "商业分析",
    "外语学习",
    "数学基础",
]

COURSE_TEMPLATES = {
    "人工智能": ["机器学习导论", "深度学习实战", "自然语言处理", "计算机视觉基础"],
    "大数据": ["Spark 大数据分析", "Hadoop 生态实践", "数据仓库建模", "实时计算入门"],
    "软件工程": ["Python Web 开发", "Java 企业级开发", "软件测试技术", "云原生应用实践"],
    "计算机基础": ["数据结构", "操作系统", "数据库系统", "计算机网络"],
    "产品设计": ["交互设计基础", "用户研究方法", "产品经理实战", "数据驱动设计"],
    "商业分析": ["商业智能分析", "统计分析基础", "营销数据分析", "金融风控建模"],
    "外语学习": ["学术英语写作", "商务英语口语", "日语入门", "英语听力提升"],
    "数学基础": ["线性代数", "概率论与数理统计", "离散数学", "最优化方法"],
}


def generate_courses(course_count: int = 80) -> pd.DataFrame:
    rows = []
    teachers = ["张华", "李敏", "王磊", "赵婷", "陈晨", "刘洋", "周宁", "孙悦"]
    for index in range(course_count):
        category = CATEGORIES[index % len(CATEGORIES)]
        base_name = COURSE_TEMPLATES[category][index % len(COURSE_TEMPLATES[category])]
        rows.append(
            {
                "course_id": f"C{index + 1:04d}",
                "course_name": f"{base_name} {index // len(CATEGORIES) + 1}",
                "category": category,
                "teacher": random.choice(teachers),
                "description": f"面向慕课学习者的{category}课程，包含案例实践与在线测验。",
            }
        )
    return pd.DataFrame(rows)


def generate_users_and_interactions(user_count: int = 240, course_count: int = 80, seed: int = 42):
    random.seed(seed)
    courses = generate_courses(course_count)
    course_by_category = {
        category: courses[courses["category"] == category]["course_id"].tolist()
        for category in CATEGORIES
    }

    users = []
    interactions = []
    for user_index in range(user_count):
        user_id = f"U{user_index + 1:05d}"
        primary = random.choice(CATEGORIES)
        secondary = random.choice([item for item in CATEGORIES if item != primary])
        target_count = random.randint(5, 14)
        selected = set()

        while len(selected) < target_count:
            category = random.choices([primary, secondary, random.choice(CATEGORIES)], weights=[0.65, 0.25, 0.10])[0]
            selected.add(random.choice(course_by_category[category]))

        users.append({"user_id": user_id, "course_order": "|".join(sorted(selected))})

        for course_id in selected:
            enroll = 1.0
            watch = random.uniform(0.0, 1.0)
            exercise = random.uniform(0.0, 1.0)
            preference = min(5.0, enroll + watch * 2.0 + exercise * 1.5)
            interactions.append(
                {
                    "user_id": user_id,
                    "course_id": course_id,
                    "enroll": 1,
                    "watch_ratio": round(watch, 3),
                    "exercise_ratio": round(exercise, 3),
                    "preference": round(preference, 3),
                }
            )

    return pd.DataFrame(users), courses, pd.DataFrame(interactions)


def main() -> None:
    ensure_directories()
    users, courses, interactions = generate_users_and_interactions()
    users.to_csv(USERS_FILE, index=False, encoding="utf-8-sig")
    courses.to_csv(COURSES_FILE, index=False, encoding="utf-8-sig")
    interactions.to_csv(INTERACTIONS_FILE, index=False, encoding="utf-8-sig")
    print(f"Sample data generated in {Path(RAW_DIR).resolve()}")
    print(f"users={len(users)}, courses={len(courses)}, interactions={len(interactions)}")


if __name__ == "__main__":
    main()
