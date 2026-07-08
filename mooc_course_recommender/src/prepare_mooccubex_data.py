import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Iterable

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import COURSES_FILE, INTERACTIONS_FILE, RAW_DIR, USERS_FILE, ensure_directories


MOOCCUBEX_DIR = RAW_DIR.parent / "mooccubex"
COURSE_URL = "https://lfs.aminer.cn/misc/moocdata/data/mooccube2/entities/course.json"
USER_URL = "https://lfs.aminer.cn/misc/moocdata/data/mooccube2/entities/user.json"


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print(f"Using cached file: {path}")
        return
    print(f"Downloading: {url}")
    with urllib.request.urlopen(url, timeout=120) as response, path.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    print(f"Saved: {path} ({path.stat().st_size / 1024 / 1024:.2f} MB)")


def download_user_sample(url: str, path: Path, max_users: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print(f"Using cached user sample: {path}")
        return
    print(f"Streaming first {max_users} users from: {url}")
    count = 0
    with urllib.request.urlopen(url, timeout=120) as response, path.open("wb") as output:
        for line in response:
            if not line.strip():
                continue
            output.write(line)
            count += 1
            if count >= max_users:
                break
    print(f"Saved sampled users: {count} rows -> {path}")


def iter_json_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as file:
        first = file.read(1)
        file.seek(0)
        if first == "[":
            data = json.load(file)
            yield from data
            return
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_course(record: dict) -> dict:
    fields = record.get("field") or record.get("fields") or []
    if isinstance(fields, str):
        fields = [fields]
    category = fields[0] if fields else "未标注"
    return {
        "course_id": record.get("id"),
        "course_name": record.get("name") or record.get("id"),
        "category": category,
        "teacher": "MOOCCubeX",
        "description": record.get("about") or record.get("prerequisites") or "",
    }


def normalize_user_course_id(course_id) -> str:
    text = str(course_id).strip()
    return text if text.startswith("C_") else f"C_{text}"


def build_raw_tables(course_path: Path, user_path: Path, max_users: int, max_interactions: int) -> None:
    ensure_directories()
    courses = [normalize_course(item) for item in iter_json_records(course_path)]
    courses_df = pd.DataFrame(courses).dropna(subset=["course_id"]).drop_duplicates("course_id")
    valid_course_ids = set(courses_df["course_id"].astype(str))

    user_rows = []
    interaction_rows = []
    for record in iter_json_records(user_path):
        user_id = record.get("id")
        course_order = record.get("course_order") or []
        if not user_id or not isinstance(course_order, list):
            continue

        selected = [
            normalize_user_course_id(course_id)
            for course_id in course_order
            if normalize_user_course_id(course_id) in valid_course_ids
        ]
        if len(selected) < 2:
            continue

        user_rows.append({"user_id": user_id, "course_order": "|".join(selected)})
        for position, course_id in enumerate(selected):
            preference = max(1.0, 3.0 - position * 0.03)
            interaction_rows.append(
                {
                    "user_id": user_id,
                    "course_id": course_id,
                    "enroll": 1,
                    "watch_ratio": "",
                    "exercise_ratio": "",
                    "preference": round(preference, 3),
                }
            )

        if len(user_rows) >= max_users or len(interaction_rows) >= max_interactions:
            break

    users_df = pd.DataFrame(user_rows)
    interactions_df = pd.DataFrame(interaction_rows)
    used_courses = set(interactions_df["course_id"].astype(str)) if not interactions_df.empty else set()
    courses_df = courses_df[courses_df["course_id"].astype(str).isin(used_courses)].copy()

    users_df.to_csv(USERS_FILE, index=False, encoding="utf-8-sig")
    courses_df.to_csv(COURSES_FILE, index=False, encoding="utf-8-sig")
    interactions_df.to_csv(INTERACTIONS_FILE, index=False, encoding="utf-8-sig")

    print("MOOCCubeX subset converted to project raw tables:")
    print(f"  users: {len(users_df)} -> {USERS_FILE}")
    print(f"  courses: {len(courses_df)} -> {COURSES_FILE}")
    print(f"  interactions: {len(interactions_df)} -> {INTERACTIONS_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert a runnable MOOCCubeX subset.")
    parser.add_argument("--max-users", type=int, default=3000)
    parser.add_argument("--max-interactions", type=int, default=30000)
    parser.add_argument("--course-url", default=COURSE_URL)
    parser.add_argument("--user-url", default=USER_URL)
    args = parser.parse_args()

    course_path = MOOCCUBEX_DIR / "course.json"
    user_sample_path = MOOCCUBEX_DIR / f"user_sample_{args.max_users}.jsonl"
    download_file(args.course_url, course_path)
    download_user_sample(args.user_url, user_sample_path, args.max_users)
    build_raw_tables(course_path, user_sample_path, args.max_users, args.max_interactions)


if __name__ == "__main__":
    main()
