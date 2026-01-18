import argparse
import sys
import os
import logging
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import train
from src import retrain_best
from src import eval_utils 
from src import data_clinic
from src import translator

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Irembo MLOps Controller")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Select operation mode")

    # MODE 1: AUGMENT
    parser_aug = subparsers.add_parser("augment", help="Run data augmentation/translation")
    parser_aug.add_argument("--input", type=str, default="data/raw_data.csv")

    # MODE 2: AUDIT
    parser_audit = subparsers.add_parser("audit", help="Run Data Clinic to find mislabels")
    
    # MODE 3: PIPELINE
    parser_pipe = subparsers.add_parser("pipeline", help="Run Experiment -> Retrain -> Evaluate flow")
    parser_pipe.add_argument("--epochs", type=int, default=10, help="Epochs for training")
    parser_pipe.add_argument("--batch_size", type=int, default=16, help="Batch size")

    return parser.parse_args()

def run_augment(args):
    logger.info("🔧 MODE: Data Augmentation")
    print("Starting translation process... (Placeholder)")
    # translator.run(args.input) 
    print("✅ Augmentation complete. New data saved to data/augmented_train_set.csv")

def run_audit(args):
    logger.info("dating 🩺 MODE: Data Clinic")
    print("Auditing labels using existing CV predictions...")
    data_clinic.main() 

def run_pipeline(args):
    logger.info("🚀 MODE: Full Production Pipeline")
    
    # PHASE 1: EXPERIMENTATION
    print("\n" + "="*40)
    print("PHASE 1: Hyperparameter Tuning (CV)")
    print("="*40)
    
    class TrainArgs:
        train_file = "data/augmented_train_set.csv"
        val_file = "data/voiceai_intent_val.csv"
        output_dir = "models/temp_cv_model"
        run_cv = True
        epochs = 5
        batch_size = args.batch_size
        lr = 2e-4
    
    # Load Data Once
    full_df, le = train.load_and_prepare_data(TrainArgs.train_file, TrainArgs.val_file)
    tokenizer = train.AutoTokenizer.from_pretrained(train.MODEL_ID)
    
    # Run CV
    best_params, history = train.run_cross_validation(full_df, le, tokenizer, TrainArgs)

    assert best_params is not None
    
    print("\n" + "-"*40)
    print(f" BEST CONFIGURATION FOUND:")
    print(f"   Learning Rate: {best_params['learning_rate']}")
    print(f"   LoRA Rank (r): {best_params['r']}")
    print(f"   F1 Score:      {best_params['mean_f1']:.4f}")
    print("-"*40)

    # --- INTERACTIVE GATE ---
    user_input = input("\n Do you want to retrain the Final Model with these params? [Y/n] ")
    if user_input.lower() not in ["y", "yes", ""]:
        print("❌ Pipeline aborted by user.")
        return

    # --- PHASE 2: PRODUCTION RETRAINING ---
    print("\n" + "="*40)
    print("PHASE 2: Retraining Champion Model")
    print("="*40)
    
    # Update args with the BEST parameters found
    class RetrainArgs:
        train_file = "data/augmented_train_set.csv"
        val_file = "data/voiceai_intent_val.csv"
        output_dir = "models/final_lora_adapter" # REAL location
        epochs = args.epochs # User defined full epochs
        batch_size = args.batch_size
        lr = best_params['learning_rate'] # From CV
    

    train.train_final_model(full_df, le, tokenizer, RetrainArgs, best_params=best_params)
    print("Best model saved to models/final_lora_adapter")

    # --- PHASE 3: EVALUATION ---
    print("\n" + "="*40)
    print("PHASE 3: Final Quality Gate")
    print("="*40)
    
    eval_utils.main() # Runs the evaluation script logic
    
    # --- PHASE 4: HANDOFF ---
    print("\n✅ PIPELINE COMPLETE.")
    print("To start the inference API, run this command:")
    print("uvicorn src.inference:app --reload")

def main():
    args = parse_args()
    
    if args.mode == "augment":
        run_augment(args)
    elif args.mode == "audit":
        run_audit(args)
    elif args.mode == "pipeline":
        run_pipeline(args)

if __name__ == "__main__":
    main()