import pandas as pd
import Levenshtein
from deep_translator import GoogleTranslator
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AugmentationPipeline:
    def __init__(self, input_file, min_confidence=0.8):
        """
        Initialize the pipeline with file path and configuration.
        """
        self.input_file = input_file
        self.min_confidence = min_confidence
        self.df = None
        self.augmented_rows = []
        self.stats = {
            'processed': 0,
            'kept': 0,
            'rejected_duplicate': 0,
            'rejected_garbage': 0,
            'rejected_too_similar': 0
        }

    def load_data(self):
        """Loads the dataset and prints initial stats."""
        try:
            self.df = pd.read_csv(self.input_file)
            logging.info(f"Loaded {len(self.df)} rows from {self.input_file}")
        except FileNotFoundError:
            logging.error(f"File not found: {self.input_file}")
            raise

    def _back_translate(self, text, lang):
        """
        Performs the back-translation: Original -> French -> Target
        Returns the augmented text or None if failed.
        """
        try:
            # Translate to French (Pivot)
            fr_text = GoogleTranslator(source='auto', target='fr').translate(text)
            
            # Translate back to Original Language (rw or en)
            # Mixed language ('mixed') defaults to English for target
            target_lang = 'rw' if lang == 'rw' else 'en'
            
            aug_text = GoogleTranslator(source='fr', target=target_lang).translate(fr_text)
            return aug_text
            
        except Exception as e:
            logging.warning(f"Translation failed for: '{text[:20]}...' | Error: {e}")
            return None

    def _validate_augmentation(self, original, augmented):
        """
        Applies Levenshtein logic to classify the quality of the augmentation.
        Returns: (status, similarity_score)
        """
        if not augmented:
            return "FAILED", 0.0

        ratio = Levenshtein.ratio(str(original), str(augmented))
        score = round(ratio, 2)

        if ratio == 1.0:
            return "DUPLICATE", score
        elif ratio > 0.95:
            return "TOO_SIMILAR", score
        elif 0.5 <= ratio <= 0.95:
            return "GOOD_PARAPHRASE", score
        elif 0.2 <= ratio < 0.5:
            return "LANGUAGE_SWAP", score
        else:
            return "GARBAGE", score

    def run(self):
        """
        Main execution method with limited printing + progress updates.
        """
        if self.df is None:
            self.load_data()
        
        if self.df is None:
             raise ValueError("Dataframe failed to load!") 

        # Filter for high-quality input data only
        high_conf_df = self.df[self.df['asr_confidence'] >= self.min_confidence]
        logging.info(f"Starting pipeline on {len(high_conf_df)} high-confidence rows...")

        print_count_success = 0
        print_count_rejected = 0
        
        for index, row in high_conf_df.iterrows():
            original_text = row['utterance_text']
            original_lang = row['language']

            # Step 1: Augment
            augmented_text = self._back_translate(original_text, original_lang)
            
            # Step 2: Validate
            status, score = self._validate_augmentation(original_text, augmented_text)
            
            # Step 3: Filter Logic
            self.stats['processed'] += 1
            
            if status in ["GOOD_PARAPHRASE", "LANGUAGE_SWAP"]:
                new_row = row.copy()
                new_row['utterance_text'] = augmented_text
                self.augmented_rows.append(new_row)
                self.stats['kept'] += 1
                
                if print_count_success < 10:
                    logging.info(f"✅ SUCCESS [{status}] Score: {score}")
                    logging.info(f"Orig: {original_text}")
                    logging.info(f"Augm: {augmented_text}")
                    logging.info("-" * 30)
                    print_count_success += 1
                
            else:
                if status == "DUPLICATE": self.stats['rejected_duplicate'] += 1
                elif status == "GARBAGE": self.stats['rejected_garbage'] += 1
                elif status == "TOO_SIMILAR": self.stats['rejected_too_similar'] += 1
                
                if print_count_rejected < 10:
                    logging.info(f"❌ REJECTED [{status}] Score: {score}")
                    logging.info(f"Orig: {original_text}")
                    logging.info(f"Augm: {augmented_text}")
                    logging.info("-" * 30)
                    print_count_rejected += 1

            # PROGRESS UPDATE (Every 50 rows)
            if self.stats['processed'] % 50 == 0:
                logging.info(f"... Processed {self.stats['processed']} rows ...")

            # Sleep to be kind to the API
            time.sleep(0.2)

        logging.info("Pipeline completed.")

    def save_results(self, output_file):
        """
        Combines original data with new high-quality augmented data and saves.
        """
        if self.df is None:
             logging.warning("No data to save!")
             return

        if not self.augmented_rows:
            logging.warning("No augmented rows were generated. Saving original only.")
            self.df.to_csv(output_file, index=False)
            return

        augmented_df = pd.DataFrame(self.augmented_rows)
        final_df = pd.concat([self.df, augmented_df], ignore_index=True)
        
        final_df.to_csv(output_file, index=False)
        logging.info(f"Saved merged dataset ({len(final_df)} rows) to {output_file}")
        
        # Print Final Summary
        print("\n" + "="*40)
        print("FINAL PIPELINE STATISTICS")
        print("="*40)
        print(f"Total Processed:      {self.stats['processed']}")
        print(f"Kept (High Quality): {self.stats['kept']}")
        print(f"Rejected (Duplicate): {self.stats['rejected_duplicate']}")
        print(f"Rejected (Too Sim.):  {self.stats['rejected_too_similar']}")
        print(f"Rejected (Garbage):   {self.stats['rejected_garbage']}")
        print("="*40)


if __name__ == "__main__":
    INPUT_FILE = 'data/voiceai_intent_train.csv'
    OUTPUT_FILE = 'data/augmented_train_set.csv'
    
    # Initialize and Run
    pipeline = AugmentationPipeline(INPUT_FILE, min_confidence=0.8)
    pipeline.run()
    pipeline.save_results(OUTPUT_FILE)