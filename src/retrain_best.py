import argparse
import json
import logging
import os
import pickle
import pandas as pd
import numpy as np
from . import eval_utils
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
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
CONFIG_PATH = "models/best_hyperparams.json"

def parse_args():
    parser = argparse.ArgumentParser(description="Retrain Final Champion Model using Inferred Hyperparameters")
    parser.add_argument("--train_file", type=str, default="data/augmented_train_set.csv", help="Path to training data")
    parser.add_argument("--val_file", type=str, default="data/voiceai_intent_val.csv", help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default="models/final_lora_adapter", help="Where to save the final model")
    return parser.parse_args()

def load_inferred_params(config_path):
    """
    Loads the best hyperparameters saved by the Nested CV process.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"CRITICAL: Configuration file {config_path} not found. Run Nested CV first.")
    
    with open(config_path, "r") as f:
        params = json.load(f)
    
    logger.info(f"Loaded Inferred Hyperparameters: {params}")
    return params

def load_and_prepare_data(train_path, val_path):
    """
    Loads data, fits LabelEncoder, and merges Train+Val for final training.
    """
    logger.info(f"Loading datasets from {train_path} and {val_path}")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Label Encoding
    le = LabelEncoder()
    le.fit(full_train_df['intent'])
    
    full_train_df['label'] = le.transform(full_train_df['intent'])
    
    os.makedirs("models", exist_ok=True)
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    logger.info(f"LabelEncoder saved to models/label_encoder.pkl. Classes: {len(le.classes_)}")
    
    return full_train_df, le

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

def main():
    args = parse_args()

    best_params = load_inferred_params(CONFIG_PATH)
    
    full_df, le = load_and_prepare_data(args.train_file, args.val_file)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenized_dataset = preprocess_dataset(full_df, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    id2label = {i: label for i, label in enumerate(le.classes_)}
    label2id = {label: i for i, label in enumerate(le.classes_)}
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, 
        num_labels=len(le.classes_), 
        id2label=id2label, 
        label2id=label2id
    )
    
    # DYNAMIC LoRA CONFIGURATION
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        inference_mode=False, 
        r=int(best_params['r']),
        lora_alpha=int(best_params['r']) * 2,
        lora_dropout=0.1, 
        target_modules=["query", "value"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Initialize Trainer with INFERRED Learning Rate
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=float(best_params['learning_rate']),
        per_device_train_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        save_strategy="no",
        report_to="none",
        logging_steps=50
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    logger.info("Starting Production Training with Inferred Parameters...")
    trainer.train()
    
    logger.info(f"Saving Champion Model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("✅ Retraining Complete. Model is ready for deployment.")

if __name__ == "__main__":
    main()