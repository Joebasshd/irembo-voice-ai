import pandas as pd
import asyncio
from deep_translator import GoogleTranslator

# Load dataset
df = pd.read_csv('data/voiceai_intent_train.csv')

async def process_translations():
    augmented_rows = []
    
    # Filter high-confidence rows
    high_conf_df = df[df['asr_confidence'] >= 0.8]
    print(f"Processing {len(high_conf_df)} rows...")

    for index, row in high_conf_df.iterrows():
        try:
            original_text = row['utterance_text']
            
            # 1. Translate to French
            # deep_translator is synchronous, so no 'await' needed usually, 
            # but it's fast enough for this script size.
            translator_to_fr = GoogleTranslator(source='auto', target='fr')
            french_text = translator_to_fr.translate(original_text)
            
            # 2. Translate back to Original (English or Kinyarwanda)
            target_lang = 'rw' if row['language'] == 'rw' else 'en'
            translator_back = GoogleTranslator(source='fr', target=target_lang)
            back_translated_text = translator_back.translate(french_text)
            
            # Save row
            new_row = row.copy()
            new_row['utterance_text'] = back_translated_text
            augmented_rows.append(new_row)
            
            if len(augmented_rows) % 10 == 0:
                print(f"Translated {len(augmented_rows)} rows...")

        except Exception as e:
            print(f"Failed on row {index}: {e}")
            continue

    return augmented_rows

if __name__ == "__main__":
    # logic to run and save...
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    augmented_data = loop.run_until_complete(process_translations())
    
    augmented_df = pd.DataFrame(augmented_data)
    final_df = pd.concat([df, augmented_df], ignore_index=True)
    final_df.to_csv('augmented_train_dataset.csv', index=False)