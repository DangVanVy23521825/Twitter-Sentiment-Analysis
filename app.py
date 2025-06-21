from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from joblib import load
import torch
import re
import numpy as np
import csv
import pandas as pd
import os

def log_history(tweet, model_name, pred_label):
    with open("history.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow([tweet, model_name, pred_label])
# === Load models ===
roberta_tokenizer = AutoTokenizer.from_pretrained("roberta_finetuned_model", local_files_only=True)
roberta_model = AutoModelForSequenceClassification.from_pretrained("roberta_finetuned_model", local_files_only=True)

twitter_tokenizer = AutoTokenizer.from_pretrained("twitter_roberta_finetuned", local_files_only=True)
twitter_model = AutoModelForSequenceClassification.from_pretrained("twitter_roberta_finetuned", local_files_only=True)

svm_model = load("svm_model/svm_tfidf_pipeline.pkl")

# === Label map ===
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# === Preprocessing functions ===
def preprocess_for_roberta(text):
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"http\S+|www\.\S+", "http", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_for_twitter_roberta(text):
    text = re.sub(r"http\S+", "HTTPURL", text)
    text = re.sub(r"@\w+", "@USER", text)
    return text

def preprocess_for_svm(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

from flask import Flask, request, jsonify, render_template
# === Flask app ===
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text")  # từ form
    model_type = request.form.get("model")

    if not text or not model_type:
        return render_template("index.html", error="Vui lòng nhập tweet và chọn mô hình.")

    if model_type == "roberta":
        text_clean = preprocess_for_roberta(text)
        inputs = roberta_tokenizer(text_clean, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = roberta_model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    elif model_type == "twitter-roberta":
        text_clean = preprocess_for_twitter_roberta(text)
        inputs = twitter_tokenizer(text_clean, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = twitter_model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    elif model_type == "svm":
        text_clean = preprocess_for_svm(text)
        pred = svm_model.predict([text_clean])[0]

    else:
        return render_template("index.html", error="Mô hình không hợp lệ.")

    label = label_map[int(pred)]
    log_history(text, model_type, label)

    return render_template("index.html", result=label, input_text=text, model=model_type)

if __name__ == "__main__":
    app.run(debug=True)
 