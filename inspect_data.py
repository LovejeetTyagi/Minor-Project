"""Quick look at the dataset before training: shape, types, missing values, balance."""
from pathlib import Path
import sys

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "dataset" / "heart.csv"


def main() -> None:
    if not DATA_PATH.exists():
        sys.exit(f"Dataset not found at {DATA_PATH}. Place heart.csv inside the dataset/ folder.")

    df = pd.read_csv(DATA_PATH)

    def section(title: str) -> None:
        print(f"\n===== {title} =====")

    section("SHAPE");            print(df.shape)
    section("COLUMNS");          print(df.columns.tolist())
    section("FIRST 5 ROWS");     print(df.head())
    section("DATA TYPES");       print(df.dtypes)
    section("MISSING VALUES");   print(df.isnull().sum())
    section("DUPLICATE ROWS");   print(df.duplicated().sum())
    section("SUMMARY STATS");    print(df.describe().T)

    if "target" in df.columns:
        section("TARGET DISTRIBUTION")
        print(df["target"].value_counts())
        print((df["target"].value_counts(normalize=True) * 100).round(1).astype(str) + " %")


if __name__ == "__main__":
    main()
