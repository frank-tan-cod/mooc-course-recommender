from pathlib import Path
import shutil

import pandas as pd
from pyspark.sql import DataFrame


def _replace_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def save_table(df: DataFrame, output_dir: Path, name: str, csv: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{name}.parquet"
    _replace_path(parquet_path)
    pdf = df.toPandas()
    pdf.to_parquet(parquet_path, index=False)
    if csv:
        csv_path = output_dir / f"{name}.csv"
        _replace_path(csv_path)
        pdf.to_csv(csv_path, index=False, encoding="utf-8-sig")


def save_pandas_table(pdf: pd.DataFrame, output_dir: Path, name: str, csv: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{name}.parquet"
    _replace_path(parquet_path)
    pdf.to_parquet(parquet_path, index=False)
    if csv:
        csv_path = output_dir / f"{name}.csv"
        _replace_path(csv_path)
        pdf.to_csv(csv_path, index=False, encoding="utf-8-sig")


def read_table(spark, base_dir: Path, name: str) -> DataFrame:
    parquet_path = base_dir / f"{name}.parquet"
    csv_path = base_dir / f"{name}.csv"
    if parquet_path.exists():
        return spark.read.parquet(str(parquet_path))
    if csv_path.exists():
        return spark.read.option("header", True).option("inferSchema", True).csv(str(csv_path))
    raise FileNotFoundError(f"Cannot find table {name} under {base_dir}")
