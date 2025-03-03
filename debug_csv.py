import pandas as pd

file_path = "data/recipes.csv"

try:
    df = pd.read_csv(file_path, delimiter=",", on_bad_lines="skip", encoding="utf-8")

    print("✅ CSV Loaded Successfully!")
    print(df.head())
except Exception as e:
    print(f"❌ Error: {e}")
