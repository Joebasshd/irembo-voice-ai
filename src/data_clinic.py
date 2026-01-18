import json
import os
import pickle
import pandas as pd
import numpy as np

PREDICTIONS_FILE = "models/cv_predictions.json"
DATA_FILE = "data/augmented_train_set.csv"
LABEL_ENCODER_FILE = "models/label_encoder.pkl"
OUTPUT_DIR = "reports"
OUTPUT_FILE = f"{OUTPUT_DIR}/potential_mislabels.csv"

def load_resources():
    """
    Loads the necessary artifacts: CV predictions, Original Data, and Label Encoder.
    """
    print("--- Loading Data Clinic Resources ---")
    
    if not os.path.exists(PREDICTIONS_FILE):
        raise FileNotFoundError(f"Could not find {PREDICTIONS_FILE}. You must run 'python src/train.py --run_cv' first.")
    
    with open(PREDICTIONS_FILE, "r") as f:
        cv_preds = json.load(f)
    print(f"Loaded {len(cv_preds)} validation predictions.")

    df = pd.read_csv(DATA_FILE)
    print(f"Loaded original dataset ({len(df)} rows).")

    with open(LABEL_ENCODER_FILE, "rb") as f:
        le = pickle.load(f)
    print("Loaded LabelEncoder.")
    
    return cv_preds, df, le

def analyze_mismatches(cv_preds, df, le):
    """
    Compares Model Confidence vs. Human Label to find suspicious rows.
    """
    print("\n--- Running Audit Logic ---")
    
    suspicious_rows = []
    
    for entry in cv_preds:
        # Map back to the original row in the CSV
        idx = entry['original_index']
        true_label_id = entry['true_label']
        pred_label_id = entry['predicted_label']
        probs = entry['probabilities']
        
        # KEY METRIC: Label Quality Score
        # This is the probability the model assigned to the HUMAN's label.
        # If this is 0.01, the model thinks the human is definitely wrong.
        label_quality_score = probs[true_label_id]
        
        # We only care if the model disagrees AND is reasonably confident in its disagreement
        # OR if the confidence in the human label is abysmal
        if (pred_label_id != true_label_id) and (label_quality_score < 0.5):
            
            row_data = df.iloc[idx]
            
            suspicious_rows.append({
                "original_index": idx,
                "text": row_data['utterance_text'],
                "human_label": le.inverse_transform([true_label_id])[0],
                "model_suggestion": le.inverse_transform([pred_label_id])[0],
                "model_confidence": round(probs[pred_label_id], 4), # Confidence in the suggestion
                "label_quality_score": round(label_quality_score, 4) # Confidence in the human label
            })
    
    return pd.DataFrame(suspicious_rows)

def main():
    # Load Everything
    try:
        cv_preds, df, le = load_resources()
    except FileNotFoundError as e:
        print(e)
        return

    # Find Mislabels
    audit_df = analyze_mismatches(cv_preds, df, le)
    
    # Sort by "Most Suspicious" (Lowest Quality Score)
    if not audit_df.empty:
        audit_df = audit_df.sort_values("label_quality_score", ascending=True)
        
        # Save Report
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        audit_df.to_csv(OUTPUT_FILE, index=False)
        
        print(f"\n✅ Audit Complete. Found {len(audit_df)} potential mislabels.")
        print(f"   Report saved to: {OUTPUT_FILE}")
        print("\nTOP 5 MOST SUSPICIOUS ROWS:")
        
        # Pretty print for the console
        print(audit_df[['text', 'human_label', 'model_suggestion', 'label_quality_score']].head(5).to_string(index=False))
    else:
        print("\nAudit Complete. No significant mislabels found!")

if __name__ == "__main__":
    main()