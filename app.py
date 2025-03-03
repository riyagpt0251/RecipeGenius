import os
import pandas as pd
import joblib
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load dataset
data_path = os.path.join("data", "recipes.csv")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Dataset not found at {os.path.abspath(data_path)}")

data = pd.read_csv(data_path, encoding="utf-8", on_bad_lines="skip")
data.columns = data.columns.str.lower()

if "ingredients" not in data.columns:
    raise KeyError(f"❌ Error: 'ingredients' column not found. Available columns: {data.columns}")

data["ingredients"] = data["ingredients"].fillna("").astype(str).str.lower()

# Vectorize ingredients
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(data["ingredients"])

num_samples = X.shape[0]  # Get number of recipes
n_neighbors = min(5, num_samples)  # Ensure n_neighbors doesn't exceed dataset size
print(f"📊 Number of recipes available: {num_samples}")

if num_samples < 2:
    raise ValueError("❌ Not enough recipes to train the model. Please add more recipes to 'data/recipes.csv'.")

# Train Nearest Neighbors model
model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
model.fit(X)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/recipe_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

# Flask App
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    recipes = None
    if request.method == "POST":
        ingredients = request.form.get("ingredients", "").lower()
        if ingredients:
            input_vec = vectorizer.transform([ingredients])
            distances, indices = model.kneighbors(input_vec)
            recipes = data.iloc[indices[0]].to_dict(orient="records")
    return render_template("index.html", recipes=recipes)

if __name__ == "__main__":
    app.run(debug=True)
