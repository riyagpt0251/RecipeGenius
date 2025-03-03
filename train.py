import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib

# Ensure the correct path
data_path = os.path.join("data", "recipes.csv")

# Check if the file exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Error: Dataset not found at {data_path}")

# Try different encodings to handle file issues
try:
    data = pd.read_csv(data_path, encoding="utf-8", sep=",", on_bad_lines="skip")
except UnicodeDecodeError:
    print("⚠️ Unicode error! Trying ISO-8859-1 encoding...")
    data = pd.read_csv(data_path, encoding="ISO-8859-1", sep=",", on_bad_lines="skip")

# Check if the dataset is empty
if data.empty:
    raise ValueError("❌ Error: The dataset is empty!")

# Ensure 'Ingredients' column exists
if 'Ingredients' not in data.columns:
    raise KeyError("❌ Error: 'Ingredients' column not found in CSV!")

# Fill missing ingredient values with empty strings
data['Ingredients'] = data['Ingredients'].fillna("").astype(str).str.lower()

# Vectorize ingredients
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Ingredients'])

# Train KNN model
model = NearestNeighbors(n_neighbors=5, algorithm='brute')
model.fit(X)

# Save the model and vectorizer
os.makedirs("model", exist_ok=True)
joblib.dump(model, os.path.join("model", "recipe_model.pkl"))
joblib.dump(vectorizer, os.path.join("model", "vectorizer.pkl"))

print("✅ Model training complete and saved!")
