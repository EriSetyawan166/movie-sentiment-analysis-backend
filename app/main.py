import os
import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load model and tokenizer
MODEL_PATH = "app/model"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.eval()

# Mapping for label interpretation
label_mapping = {0: "Negative", 1: "Positive"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Invalid input, 'text' key is required"}), 400

        input_text = data["text"]
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence, label_idx = torch.max(probabilities, dim=1)
            confidence = confidence.item() * 100 
            label = label_mapping[label_idx.item()]
        
        response = {
            "data": {
                "confidence": f"{confidence:.2f}%",
                "input_text": input_text,
                "label": label
            },
            "model_version": "1.0.0",
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200
