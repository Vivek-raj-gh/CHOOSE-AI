from flask import Flask, jsonify
import pandas as pd
import os

app = Flask(__name__)

# Get absolute path of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build correct path to CSV
DATA_PATH = os.path.join(BASE_DIR, "..", "electronics.csv")

df = pd.read_csv(DATA_PATH)

@app.route("/api/products", methods=["GET"])
def get_products():
    return jsonify(df.to_dict(orient="records"))

@app.route("/api/products/<category>", methods=["GET"])
def get_products_by_category(category):
    filtered = df[df["category"].str.lower() == category.lower()]
    return jsonify(filtered.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
