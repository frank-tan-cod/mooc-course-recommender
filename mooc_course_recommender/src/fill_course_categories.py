from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import OUTPUT_DIR, PROCESSED_DIR, RAW_DIR


UNLABELED = "未标注"
REPORT_FILE = PROCESSED_DIR / "category_fill_report.csv"
FILLED_REPORT_OLD_CATEGORY = "未标注"

BIG_CATEGORY_BY_EXISTING = {
    "哲学": "哲学",
    "理论经济学": "经济学",
    "应用经济学": "经济学",
    "法学": "法学",
    "政治学": "法学",
    "社会学": "法学",
    "民族学": "法学",
    "教育学": "教育学",
    "心理学": "教育学",
    "体育学": "教育学",
    "中国语言文学": "文学",
    "外国语言文学": "文学",
    "新闻传播学": "文学",
    "历史学": "历史学",
    "科学技术史": "历史学",
    "数学": "理学",
    "物理学": "理学",
    "化学": "理学",
    "生物学": "理学",
    "地理学": "理学",
    "地质学": "理学",
    "系统科学": "理学",
    "大气科学": "理学",
    "力学": "理学",
    "机械工程": "工学",
    "光学工程": "工学",
    "仪器科学与技术": "工学",
    "材料科学与工程": "工学",
    "电子科学与技术": "工学",
    "信息与通信工程": "工学",
    "控制科学与工程": "工学",
    "计算机科学与技术": "工学",
    "建筑学": "工学",
    "土木工程": "工学",
    "水利工程": "工学",
    "测绘科学与技术": "工学",
    "化学工程与技术": "工学",
    "地质资源与地质工程": "工学",
    "石油与天然气工程": "工学",
    "交通运输工程": "工学",
    "船舶与海洋工程": "工学",
    "轻工技术与工程": "工学",
    "纺织科学与工程": "工学",
    "环境科学与工程": "工学",
    "食品科学与工程": "工学",
    "生物医学工程": "工学",
    "电气工程": "工学",
    "动力工程及工程热物理": "工学",
    "作物学": "农学",
    "园艺学": "农学",
    "兽医学": "农学",
    "基础医学": "医学",
    "临床医学": "医学",
    "口腔医学": "医学",
    "公共卫生与预防医学": "医学",
    "药学": "医学",
    "管理科学与工程": "管理学",
    "工商管理": "管理学",
    "公共管理": "管理学",
    "农林经济管理": "管理学",
    "情报与档案管理": "管理学",
    "艺术学": "艺术学",
    "军事思想及军事历史": "军事学",
    "战略学": "军事学",
    "战术学": "军事学",
    "军队政治工作学": "军事学",
}

KEYWORDS = [
    ("军事学", ["军事", "国防", "战略", "战术", "航母", "舰载机", "军队", "武器", "战争", "作战"]),
    (
        "医学",
        [
            "医学",
            "临床",
            "护理",
            "药物",
            "药学",
            "药理",
            "免疫",
            "病原",
            "传染病",
            "疾病",
            "诊断",
            "治疗",
            "生理",
            "解剖",
            "胚胎",
            "血液",
            "肿瘤",
            "急诊",
            "流行病",
            "公共卫生",
            "预防医学",
            "生殖",
            "影像学",
            "医事",
            "医学生",
            "健康",
            "营养学",
            "康复",
        ],
    ),
    ("农学", ["农业", "农学", "园艺", "作物", "畜牧", "兽医", "植物保护", "林业", "茶树", "水产"]),
    (
        "艺术学",
        [
            "艺术",
            "美术",
            "美育",
            "音乐",
            "钢琴",
            "声乐",
            "歌唱",
            "舞蹈",
            "戏剧",
            "电影",
            "影视",
            "摄影",
            "书法",
            "篆刻",
            "水墨",
            "绘画",
            "版画",
            "广告设计",
            "动画",
            "数字媒体",
            "化妆",
            "造型",
            "服装",
            "室内乐",
            "山水写生",
            "漆艺",
            "Lacquer",
        ],
    ),
    (
        "管理学",
        [
            "管理",
            "会计",
            "财务",
            "营销",
            "客户关系",
            "人力资源",
            "电子商务",
            "电商",
            "物流",
            "创业",
            "企业",
            "商业模式",
            "运营",
            "质量管理",
            "项目管理",
            "行政职业",
            "工商",
            "组织行为",
            "供应链",
            "领导力",
            "创新管理",
            "公共管理",
            "档案",
            "图书馆",
            "旅游管理",
            "酒店管理",
            "Hospitality",
        ],
    ),
    ("经济学", ["经济学", "经济", "金融", "证券", "投资", "贸易", "财政", "税收", "保险", "产业组织", "计量经济", "宏观", "微观", "市场机制", "资本市场"]),
    ("法学", ["法律", "法学", "知识产权", "民法", "刑法", "宪法", "行政法", "诉讼", "司法", "犯罪", "诈骗", "骗局", "政治", "国际关系", "社会主义", "马克思主义", "思想道德", "社会学", "社会创新", "社会企业"]),
    ("教育学", ["教育", "教学", "教师", "课程思政", "学习科学", "学前", "幼儿", "心理", "人格", "大学新生", "考试", "导师", "人才培养", "在线教学", "教学论", "慕课"]),
    ("文学", ["文学", "语言", "英语", "English", "翻译", "笔译", "写作", "汉语", "国文", "语文", "诗", "古诗", "经典导读", "名著", "新闻", "传播", "广告", "媒体", "文化传播", "应用写作", "语音语调", "四六级", "四级", "六级"]),
    ("历史学", ["历史", "史学", "考古", "资治通鉴", "隋唐", "五代", "科举", "佛教", "民俗", "民间工艺", "传统文化", "校园文化"]),
    ("哲学", ["哲学", "伦理", "道德经", "易经", "儒", "佛", "道家", "人生修养", "美学", "逻辑学", "智慧启示"]),
    ("理学", ["数学", "微积分", "高等数学", "线性代数", "概率", "统计", "矩阵", "数值分析", "复变函数", "离散数学", "计算几何", "科学计算", "小波", "物理", "力学", "光学", "热力学", "化学", "生物学", "地理", "地质", "大气科学", "天文", "科学史"]),
    ("工学", ["计算机", "程序", "编程", "C语言", "Python", "Java", "Web", "HTML", "CSS", "网页制作", "数据库", "数据结构", "操作系统", "编译", "算法", "软件", "网络", "信息安全", "病毒分析", "人工智能", "机器学习", "深度学习", "数据挖掘", "大数据", "数据科学", "数据可视化", "云计算", "云服务", "AWS", "Lambda", "API Gateway", "物联网", "区块链", "5G", "通信", "电子", "电路", "电工", "单片机", "传感器", "遥感", "图像识别", "智能语音", "自动化", "控制", "系统理论", "机械", "工程制图", "建筑", "土木", "道路", "交通", "环境工程", "水利", "测绘", "摄影测量", "材料", "能源", "太阳能", "化工", "石油", "采矿", "矿井", "供暖", "给水", "排水", "泵", "压缩机", "供配电", "输电", "钢结构", "混凝土", "结构抗震", "模具", "塑料", "制药设备", "工业炉", "仪器", "SoC", "芯片", "FPGA", "IC", "光电", "集成电路", "信号与系统", "互换性", "测量技术", "检测技术", "测试技术", "试验设计", "优化设计", "样机", "航空", "航天", "飞行", "飞动力学", "电磁场", "热流体", "轮机", "船舶", "Aerospace", "Circuit"]),
]

OVERRIDES = [
    ("财务管理", "管理学"),
    ("管理会计", "管理学"),
    ("会计学", "管理学"),
    ("营销", "管理学"),
    ("客户关系管理", "管理学"),
    ("人力资源管理", "管理学"),
    ("电子商务", "管理学"),
    ("创新创业", "管理学"),
    ("创业", "管理学"),
    ("物流", "管理学"),
    ("质量管理", "管理学"),
    ("配送中心设计与管理", "管理学"),
    ("商业计划书", "管理学"),
    ("大学计算机", "工学"),
    ("计算机组成", "工学"),
    ("数据结构", "工学"),
    ("编译原理", "工学"),
    ("软件工程", "工学"),
    ("程序设计", "工学"),
    ("软件设计", "工学"),
    ("面向对象分析与设计", "工学"),
    ("电子系统设计", "工学"),
    ("电子设计", "工学"),
    ("模拟电子技术", "工学"),
    ("现代电子系统", "工学"),
    ("数字集成电路", "工学"),
    ("集成电路", "工学"),
    ("电路", "工学"),
    ("FPGA", "工学"),
    ("IC设计", "工学"),
    ("光电仪器设计", "工学"),
    ("机械设计", "工学"),
    ("结构抗震设计", "工学"),
    ("钢结构设计", "工学"),
    ("结构现代设计", "工学"),
    ("架空输电线路设计", "工学"),
    ("模具设计", "工学"),
    ("产品设计与开发", "工学"),
    ("模块化产品设计", "工学"),
    ("数字化产品设计", "工学"),
    ("装备自动化工程设计", "工学"),
    ("飞行器综合设计", "工学"),
    ("导弹总体设计", "军事学"),
    ("试验设计与分析", "理学"),
    ("随机过程", "理学"),
    ("优化设计", "工学"),
    ("Solar Energy", "工学"),
    ("摄影测量", "工学"),
    ("现代遥感", "工学"),
    ("传感器", "工学"),
    ("网页制作", "工学"),
    ("AI技能初体验", "工学"),
    ("图像识别", "工学"),
    ("智能语音", "工学"),
    ("智能车", "工学"),
    ("AWS", "工学"),
    ("Lambda", "工学"),
    ("API Gateway", "工学"),
    ("云服务", "工学"),
    ("化工系统工程", "工学"),
    ("化工单元", "工学"),
    ("印刷色彩", "工学"),
    ("纺织导论", "工学"),
    ("焙烤食品加工", "工学"),
    ("采矿学", "工学"),
    ("矿井通风", "工学"),
    ("泵与泵站", "工学"),
    ("供暖工程", "工学"),
    ("水处理工程", "工学"),
    ("给水排水", "工学"),
    ("供配电技术", "工学"),
    ("环境监测", "工学"),
    ("互换性", "工学"),
    ("技术测量", "工学"),
    ("测量技术", "工学"),
    ("制药设备", "工学"),
    ("工业炉窑", "工学"),
    ("新型染整设备", "工学"),
    ("水下声信道", "工学"),
    ("时域测试技术", "工学"),
    ("基础工程学", "工学"),
    ("工程图学", "工学"),
    ("房屋建筑学", "工学"),
    ("居住区规划", "工学"),
    ("风景园林", "农学"),
    ("园林", "农学"),
    ("线性系统理论", "工学"),
    ("电路分析", "工学"),
    ("Fundamentals of Circuit", "工学"),
    ("考试不挂科系列课-4小时学完高等数学", "理学"),
    ("考试不挂科系列课-4小时学完线性代数", "理学"),
    ("考试不挂科系列课-2小时学完大学物理", "理学"),
    ("考试不挂科系列课-2.5小时学完大学物理", "理学"),
    ("四级", "文学"),
    ("六级", "文学"),
    ("基础笔译", "文学"),
    ("实用英语", "文学"),
    ("Hospitality English", "文学"),
    ("食品营养学", "医学"),
    ("教学设计", "教育学"),
    ("课程设计", "教育学"),
    ("微课设计", "教育学"),
    ("翻转课堂", "教育学"),
    ("在线学习活动", "教育学"),
    ("武术", "教育学"),
    ("短跑", "教育学"),
    ("定向运动", "教育学"),
    ("运动训练", "教育学"),
    ("视觉传达", "艺术学"),
    ("艺术设计", "艺术学"),
    ("环境艺术设计", "艺术学"),
    ("动画美术设计", "艺术学"),
    ("广告设计", "艺术学"),
    ("平面广告", "艺术学"),
    ("平面设计", "艺术学"),
    ("版式设计", "艺术学"),
    ("图案设计", "艺术学"),
    ("服装", "艺术学"),
    ("时装", "艺术学"),
    ("鞋类设计", "艺术学"),
    ("舞台服装", "艺术学"),
    ("UI设计", "艺术学"),
    ("交互设计", "艺术学"),
    ("网页创意与艺术设计", "艺术学"),
    ("设计素描", "艺术学"),
    ("设计美学", "艺术学"),
    ("设计学原理", "艺术学"),
    ("设计思维", "艺术学"),
    ("设计之美", "艺术学"),
    ("设计的力量", "艺术学"),
    ("产品设计原理", "艺术学"),
    ("交通工具设计", "艺术学"),
    ("家具设计", "艺术学"),
]

PRIORITY = ["医学", "军事学", "农学", "工学", "理学", "管理学", "经济学", "法学", "教育学", "文学", "历史学", "哲学", "艺术学"]
PRIORITY_INDEX = {name: index for index, name in enumerate(PRIORITY)}

ENGINEERING_CONTEXT = [
    "系统",
    "电子",
    "电路",
    "机械",
    "工程",
    "结构",
    "材料",
    "设备",
    "仪器",
    "自动化",
    "软件",
    "程序",
    "计算机",
    "网络",
    "芯片",
    "SoC",
    "FPGA",
    "IC",
    "光电",
    "模具",
    "矿",
    "能源",
    "供配电",
    "输电",
    "建筑",
    "水处理",
    "给水",
    "排水",
    "测试",
    "测量",
    "检测",
]

ART_DESIGN_CONTEXT = [
    "艺术设计",
    "视觉传达",
    "服装设计",
    "时装",
    "广告设计",
    "平面设计",
    "版式设计",
    "图案设计",
    "动画",
    "美术",
    "舞台",
    "室内",
    "景观",
    "UI",
    "交互设计",
    "家具设计",
    "产品设计原理",
    "设计美学",
    "设计素描",
    "设计史",
    "设计学",
]


def contains(text: str, key: str) -> bool:
    return key.lower() in text.lower()


def classify(row: dict[str, str], inherited_by_name: dict[str, Counter]) -> tuple[str, str]:
    name = (row.get("course_name") or "").strip()
    desc = (row.get("description") or "").strip()
    text = f"{name} {desc}"

    for key, category in OVERRIDES:
        if contains(name, key):
            return category, f"override:{key}"

    if contains(name, "设计") or contains(desc, "设计"):
        engineering_hits = [key for key in ENGINEERING_CONTEXT if contains(text, key)]
        if engineering_hits:
            return "工学", "design_context_engineering:" + "|".join(engineering_hits[:5])
        art_hits = [key for key in ART_DESIGN_CONTEXT if contains(text, key)]
        if art_hits:
            return "艺术学", "design_context_art:" + "|".join(art_hits[:5])

    inherited = inherited_by_name.get(name)
    if inherited:
        detailed = inherited.most_common(1)[0][0]
        return BIG_CATEGORY_BY_EXISTING.get(detailed, detailed), f"same_name:{detailed}"

    scores: Counter[str] = Counter()
    hits: defaultdict[str, list[str]] = defaultdict(list)
    for category, keywords in KEYWORDS:
        for keyword in keywords:
            if contains(name, keyword):
                scores[category] += 3
                hits[category].append(keyword)
            elif contains(desc, keyword):
                scores[category] += 1
                hits[category].append(keyword)

    if scores:
        best = sorted(scores, key=lambda item: (-scores[item], PRIORITY_INDEX.get(item, 99), item))[0]
        return best, "keywords:" + "|".join(hits[best][:5])

    return "教育学", "fallback:generic_course"


def read_raw_courses(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_raw_courses(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["course_id", "course_name", "category", "teacher", "description"])
        writer.writeheader()
        writer.writerows(rows)


def load_filled_course_ids() -> set[str]:
    if not REPORT_FILE.exists():
        return set()
    report = pd.read_csv(REPORT_FILE)
    if "course_id" not in report.columns:
        return set()
    return set(report["course_id"].dropna().astype(str))


def update_table_category(base_dir: Path, name: str, course_category: dict[str, str]) -> int | None:
    csv_path = base_dir / f"{name}.csv"
    parquet_path = base_dir / f"{name}.parquet"
    if not csv_path.exists() and not parquet_path.exists():
        return None

    df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.read_csv(csv_path)
    if "course_id" in df.columns and "category" in df.columns:
        mapped = df["course_id"].map(course_category)
        df["category"] = mapped.fillna(df["category"])

    if csv_path.exists():
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    if parquet_path.exists():
        df.to_parquet(parquet_path, index=False)
    return len(df)


def save_courses_clean(course_rows: list[dict[str, str]]) -> int:
    df = pd.DataFrame(course_rows)[["course_id", "course_name", "category", "teacher", "description"]]
    df = df.drop_duplicates("course_id")
    df.to_csv(PROCESSED_DIR / "courses_clean.csv", index=False, encoding="utf-8-sig")
    df.to_parquet(PROCESSED_DIR / "courses_clean.parquet", index=False)
    return len(df)


def refresh_category_distribution(course_rows: list[dict[str, str]]) -> int | None:
    csv_path = PROCESSED_DIR / "interactions_clean.csv"
    parquet_path = PROCESSED_DIR / "interactions_clean.parquet"
    if not csv_path.exists() and not parquet_path.exists():
        return None

    interactions = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.read_csv(csv_path)
    course_meta = pd.DataFrame(course_rows)[["course_id", "category"]].drop_duplicates("course_id")
    distribution = interactions[["course_id"]].merge(course_meta, on="course_id", how="left")
    distribution["category"] = distribution["category"].fillna("未知")
    distribution = (
        distribution.groupby("category", as_index=False)
        .agg(course_count=("course_id", "nunique"), interaction_count=("course_id", "size"))
        .sort_values("interaction_count", ascending=False)
    )
    distribution.to_csv(PROCESSED_DIR / "category_distribution.csv", index=False, encoding="utf-8-sig")
    distribution.to_parquet(PROCESSED_DIR / "category_distribution.parquet", index=False)
    return len(distribution)


def refresh_recommendation_metrics(course_rows: list[dict[str, str]]) -> dict[str, float] | None:
    recommendations_path = OUTPUT_DIR / "recommendations.parquet"
    latest_metrics_path = OUTPUT_DIR / "latest_metrics.parquet"
    history_path = OUTPUT_DIR / "metrics_history.parquet"
    if not recommendations_path.exists() or not latest_metrics_path.exists():
        return None

    recommendations = pd.read_parquet(recommendations_path)
    if recommendations.empty:
        metrics = {"coverage": 0.0, "diversity": 0.0, "popular_recommendation_ratio": 0.0}
    else:
        total_course_count = pd.DataFrame(course_rows)["course_id"].nunique()
        recommended_course_count = recommendations["course_id"].nunique()
        coverage = recommended_course_count / total_course_count if total_course_count else 0.0
        diversity = recommendations.groupby("user_id")["category"].nunique().mean()
        course_freq = recommendations.groupby("course_id").size()
        threshold = course_freq.quantile(0.8)
        popular_count = course_freq[course_freq >= threshold].sum()
        metrics = {
            "coverage": round(float(coverage), 4),
            "diversity": round(float(diversity), 4),
            "popular_recommendation_ratio": round(float(popular_count / len(recommendations)), 4),
        }

    latest = pd.read_parquet(latest_metrics_path)
    for key, value in metrics.items():
        if key in latest.columns:
            latest.loc[:, key] = value
    latest.to_csv(OUTPUT_DIR / "latest_metrics.csv", index=False, encoding="utf-8-sig")
    latest.to_parquet(latest_metrics_path, index=False)

    if history_path.exists() and "trained_at" in latest.columns:
        history = pd.read_parquet(history_path)
        trained_at = latest["trained_at"].iloc[0]
        mask = history["trained_at"] == trained_at
        for key, value in metrics.items():
            if key in history.columns:
                history.loc[mask, key] = value
        history.to_csv(OUTPUT_DIR / "metrics_history.csv", index=False, encoding="utf-8-sig")
        history.to_parquet(history_path, index=False)

    return metrics


def main() -> None:
    raw_courses_file = RAW_DIR / "courses.csv"
    rows = read_raw_courses(raw_courses_file)

    filled_ids = load_filled_course_ids()
    inherited_by_name: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        category = (row.get("category") or "").strip()
        if category and category != UNLABELED and row.get("course_id") not in filled_ids:
            inherited_by_name[(row.get("course_name") or "").strip()][category] += 1

    report_rows = []
    before = Counter((row.get("category") or "").strip() for row in rows)
    for row in rows:
        if (row.get("category") or "").strip() == UNLABELED or row.get("course_id") in filled_ids:
            old_category = (row.get("category") or "").strip() or UNLABELED
            new_category, reason = classify(row, inherited_by_name)
            row["category"] = new_category
            report_rows.append(
                {
                    "course_id": row["course_id"],
                    "course_name": row["course_name"],
                    "old_category": FILLED_REPORT_OLD_CATEGORY if row.get("course_id") in filled_ids else old_category,
                    "new_category": new_category,
                    "reason": reason,
                }
            )

    write_raw_courses(raw_courses_file, rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if report_rows or not REPORT_FILE.exists():
        pd.DataFrame(report_rows).to_csv(REPORT_FILE, index=False, encoding="utf-8-sig")

    course_category = {row["course_id"]: row["category"] for row in rows}
    updated_tables = {}
    updated_tables["processed/courses_clean"] = save_courses_clean(rows)
    for table in ["popular_courses", "course_learn_counts"]:
        count = update_table_category(PROCESSED_DIR, table, course_category)
        if count is not None:
            updated_tables[f"processed/{table}"] = count

    count = refresh_category_distribution(rows)
    if count is not None:
        updated_tables["processed/category_distribution"] = count

    count = update_table_category(OUTPUT_DIR, "recommendations", course_category)
    if count is not None:
        updated_tables["output/recommendations"] = count

    metrics = refresh_recommendation_metrics(rows)
    if metrics is not None:
        updated_tables["output/latest_metrics"] = len(metrics)

    after = Counter((row.get("category") or "").strip() for row in rows)
    print(f"raw rows: {len(rows)}")
    print(f"unlabeled before: {before.get(UNLABELED, 0)}")
    print(f"filled unlabeled: {len(report_rows)}")
    print(f"unlabeled after: {after.get(UNLABELED, 0)}")
    print("filled category counts:")
    for category, count in Counter(row["new_category"] for row in report_rows).most_common():
        print(f"  {category}: {count}")
    print("updated tables:")
    for table, count in updated_tables.items():
        print(f"  {table}: {count} rows")
    if metrics is not None:
        print("refreshed recommendation metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    print(f"report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
