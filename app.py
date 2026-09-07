"""
Serves the trained Iris classifier as a REST API.
Demonstrates: model deployment / model serving — the "Ops" half of MLOps.
"""
import os

import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = "model/model.joblib"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


@app.route("/health", methods=["GET"])
def health_check():
    status = "ok" if model is not None else "model_not_loaded"
    return jsonify({"status": status}), 200 if model is not None else 503


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "model not loaded — run train.py first"}), 503

    data = request.get_json(silent=True)
    if not data or "features" not in data:
        return jsonify({
            "error": "Send JSON like: {\"features\": [5.1, 3.5, 1.4, 0.2]} "
                     "(sepal_length, sepal_width, petal_length, petal_width)"
        }), 400

    features = data["features"]
    if len(features) != 4:
        return jsonify({"error": "expected exactly 4 features"}), 400

    try:
        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]
    except Exception as e:
        return jsonify({"error": f"prediction failed: {str(e)}"}), 400

    return jsonify({
        "prediction": CLASS_NAMES[prediction],
        "confidence": round(float(max(probabilities)), 4),
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
