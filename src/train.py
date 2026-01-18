import argparse
import os
import json
import logging
import pickle
import pandas as pd
import numpy as np
import evaluate
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_ID = "davlan/afro-xlmr-small"
MAX_LENGTH = 128

def parse_args():
    parser = argparse.ArgumentParser(description="Train AfroXLMR-LoRA for Irembo Intent Classification")
    parser.add_argument("--train_file", type=str, default="data/augmented_train_set.csv", help="Path to training data")
    parser.add_argument("--val_file", type=str, default="data/voiceai_intent_val.csv", help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default="models/final_lora_adapter", help="Where to save the final model")
    parser.add_argument("--run_cv", action="store_true", help="If set, runs 5-Fold CV before final training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    return parser.parse_args()

def load_and_prepare_data(train_path, val_path):
    """
    Loads data, fits LabelEncoder, and merges Train+Val for final training.
    """
    logger.info(f"Loading datasets from {train_path} and {val_path}")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)
    
    le = LabelEncoder()
    le.fit(full_train_df['intent'])
    
    full_train_df['label'] = le.transform(full_train_df['intent'])
    
    # Save LabelEncoder for inference/evaluation later
    os.makedirs("models", exist_ok=True)
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    logger.info(f"LabelEncoder saved to models/label_encoder.pkl. Classes: {len(le.classes_)}")
    
    return full_train_df, le

def get_model(num_labels, id2label, label2id, r=16):
    """
    Returns a fresh AfroXLMR model with LoRA Config.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, 
        num_labels=num_labels, 
        id2label=id2label, 
        label2id=label2id
    )
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        inference_mode=False, 
        r=r, 
        lora_alpha=r * 2,
        lora_dropout=0.1, 
        target_modules=["query", "value"]
    )
    return get_peft_model(model, peft_config)

def compute_metrics(eval_pred):
    metric = evaluate.load("f1")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels, average="macro")


def preprocess_dataset(df, tokenizer):
    """
    Converts Pandas DF to Hugging Face Dataset, Tokenizes, and Cleans columns.
    """
    ds = Dataset.from_pandas(df)
    
    def tokenize_fn(examples):
        return tokenizer(examples["utterance_text"], truncation=True, padding=True, max_length=MAX_LENGTH)
    
    tokenized_ds = ds.map(tokenize_fn, batched=True)
    
    keep_cols = ["input_ids", "attention_mask", "label"]
    remove_cols = [col for col in tokenized_ds.column_names if col not in keep_cols]
    
    tokenized_ds = tokenized_ds.remove_columns(remove_cols)
    return tokenized_ds

def run_cross_validation(df, le, tokenizer, args):
    """
    Performs hyperparameter tuning using 5-Fold CV and returns the best parameters.
    Saves out-of-sample predictions for the best configuration to models/cv_predictions.json
    """
    logger.info("--- Starting Hyperparameter Tuning with 5-Fold Cross-Validation ---")
    
    # Define hyperparameter grid
    lrs = [1e-4, 2e-4, 5e-4]
    rs = [8, 16, 32]
    batch_size = 16
    
    best_params = None
    best_score = -1
    best_params_history = []
    
    # Variable to store predictions of the currently winning configuration
    best_config_predictions = [] 
    
    id2label = {i: label for i, label in enumerate(le.classes_)}
    label2id = {label: i for i, label in enumerate(le.classes_)}
    
    for lr in lrs:
        for r in rs:
            logger.info(f"Tuning lr={lr}, r={r}")
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_scores = []
            
            # Temporary list to hold predictions for THIS specific configuration
            current_config_predictions = []
            
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
                logger.info(f"Processing Fold {fold + 1}/5 for lr={lr}, r={r}...")
                
                train_fold = df.iloc[train_idx]
                val_fold = df.iloc[val_idx]
                
                tokenized_train = preprocess_dataset(train_fold, tokenizer)
                tokenized_val = preprocess_dataset(val_fold, tokenizer)
                
                model = get_model(len(le.classes_), id2label, label2id, r=r)
                
                training_args = TrainingArguments(
                    output_dir=f"results_cv_lr{lr}_r{r}_fold_{fold}",
                    learning_rate=lr,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    num_train_epochs=5,  # Reduced epochs for CV speed
                    weight_decay=0.01,
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    load_best_model_at_end=True,
                    report_to="none",
                    logging_steps=50
                )
                
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=tokenized_train,
                    eval_dataset=tokenized_val,
                    processing_class=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics, # type: ignore
                )
                
                trainer.train()
                metrics = trainer.evaluate()
                fold_scores.append(metrics['eval_f1'])
                logger.info(f"Fold {fold+1} F1: {metrics['eval_f1']:.4f}")
                
                # Predict on the validation fold to get "clean" probabilities for data_clinic
                predictions_output = trainer.predict(tokenized_val) # type: ignore
                logits = predictions_output.predictions
                
                exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                
                # Store indices and probs for this fold
                for i, idx in enumerate(val_idx):
                    current_config_predictions.append({
                        "original_index": int(idx),
                        "predicted_label": int(np.argmax(probs[i])),
                        "true_label": int(df.iloc[idx]['label']),
                        "probabilities": probs[i].tolist()
                    })
            
            mean_f1 = np.mean(fold_scores)
            std_f1 = np.std(fold_scores)
            logger.info(f"Config lr={lr}, r={r}: Mean F1 = {mean_f1:.4f} +/- {std_f1:.4f}")
            
            params = {'learning_rate': lr, 'r': r, 'mean_f1': mean_f1, 'std_f1': std_f1}
            best_params_history.append(params)
            
            # If this is the new best model, save its predictions
            if mean_f1 > best_score:
                best_score = mean_f1
                best_params = params
                best_config_predictions = current_config_predictions
    
    logger.info(f"Best Hyperparameters: {best_params}")
    
    # SAVE PREDICTIONS FOR DATA CLINIC
    if best_config_predictions:
        preds_file = "models/cv_predictions.json"
        os.makedirs("models", exist_ok=True)
        with open(preds_file, "w") as f:
            json.dump(best_config_predictions, f)
        logger.info(f"✅ Saved {len(best_config_predictions)} out-of-sample predictions to {preds_file} for Data Clinic.")
    
    return best_params, best_params_history

def train_final_model(df, le, tokenizer, args, best_params=None):
    """
    Trains the final model on ALL data (Train + Val) and saves it.
    Uses best_params if provided.
    """
    logger.info("--- Starting Final Model Training (Champion) ---")
    
    id2label = {i: label for i, label in enumerate(le.classes_)}
    label2id = {label: i for i, label in enumerate(le.classes_)}
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Full Dataset
    tokenized_full = preprocess_dataset(df, tokenizer)
    r = best_params['r'] if best_params else 16
    
    model = get_model(len(le.classes_), id2label, label2id, r=r)
    
    # Use best_params if available, else args
    lr = best_params['learning_rate'] if best_params else args.lr
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_strategy="no", # We save manually at the end
        report_to="none",
        logging_steps=50
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_full,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    logger.info(f"Saving final adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

def main():
    args = parse_args()
    
    # Load Data & Resources
    full_df, le = load_and_prepare_data(args.train_file, args.val_file)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    best_params = None
    best_params_history = []

    # Run CV with Hyperparameter Tuning
    if args.run_cv:
        best_params, best_params_history = run_cross_validation(full_df, le, tokenizer, args)
        
        # Save best_params_history to JSON for retrain_best.py
        os.makedirs("models", exist_ok=True)
        with open("models/best_hyperparams.json", "w") as f:
            json.dump(best_params_history, f)
        logger.info(f"Best hyperparameters history saved to models/best_hyperparams.json")
    
    # 2. Train Final Model
    train_final_model(full_df, le, tokenizer, args, best_params)

if __name__ == "__main__":
    main()