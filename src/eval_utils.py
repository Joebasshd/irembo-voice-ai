import os
import torch
import pickle
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from datasets import Dataset

# Constants
BASE_MODEL_ID = "davlan/afro-xlmr-small"
ADAPTER_PATH = "models/final_lora_adapter"
LE_PATH = "models/label_encoder.pkl"
TEST_DATA_PATH = "data/voiceai_intent_test.csv"
REPORTS_DIR = "reports"

def load_resources():
    print("--- Loading Evaluation Resources ---")
    
    with open(LE_PATH, "rb") as f:
        le = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID, 
        num_labels=len(le.classes_)
    )
    
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    
    print(f"Resources loaded. Evaluating {len(le.classes_)} intents.")
    return model, tokenizer, le

def run_inference(model, tokenizer, texts, device="cpu"):
    """
    Performs inference and tracks timing for latency reporting.
    """
    model.to(device)
    predictions = []
    latencies = []
    
    print(f"Running inference on {len(texts)} samples...")
    
    for text in texts:
        start_time = time.time()
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=-1).item()
            predictions.append(pred_id)
            
        latencies.append(time.time() - start_time)
        
    return predictions, np.mean(latencies)

def generate_reports(test_df, y_pred, le, avg_latency):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    y_true = test_df['label'].values
    target_names = list(le.classes_)

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("\n--- Classification Report ---")

    with open("reports/report.txt", "w") as file:
        if isinstance(report, dict):
            file.write(json.dumps(report, indent=4))  # Convert dict to JSON string
        else:
            file.write(report)  # Already a string

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
    plt.title("Intent Classification Confusion Matrix")
    plt.ylabel('Actual Intent')
    plt.xlabel('Predicted Intent')
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/confusion_matrix.png")
    print(f"Confusion matrix saved to {REPORTS_DIR}/")

    # Error Analysis (The Wall of Shame)
    test_df['predicted_intent'] = le.inverse_transform(y_pred)
    test_df['actual_intent'] = le.inverse_transform(y_true)
    
    errors = test_df[test_df['predicted_intent'] != test_df['actual_intent']]
    errors.to_csv(f"{REPORTS_DIR}/misclassifications.csv", index=False)
    print(f"Error log saved ({len(errors)} misclassifications).")

def main():
    # Load test data
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    # Load model and mapping
    model, tokenizer, le = load_resources()
    
    # Pre-encode labels in the test set for comparison
    test_df['label'] = le.transform(test_df['intent'])
    
    # Run Predictions
    y_pred, avg_latency = run_inference(model, tokenizer, test_df['utterance_text'].tolist())
    
    # Generate and save reports
    generate_reports(test_df, y_pred, le, avg_latency)

if __name__ == "__main__":
    main()