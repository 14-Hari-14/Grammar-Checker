import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
from utils import GrammarChecker

# Configuration
CONFIG = {
    "model_path": "grammar_model.pkl",
    "test_csv": "/home/hari/shl_dataset/test.csv",
    "audio_dir": "/home/hari/shl_dataset/audios/test",
    "output_file": "submission_new.csv",
    "progress_file": "predict_progress_new.json",
    "checkpoint_interval": 5,  # Save every 5 files
    "corrupted_score": 1.5  # Low score for corrupted files
}

def load_progress():
    # Load progress from previous run
    try:
        with open(CONFIG['progress_file'], 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "processed": [],
            "results": []
        }

def save_progress(progress):
    # Save current progress
    with open(CONFIG['progress_file'], 'w') as f:
        json.dump(progress, f)

def is_corrupted_file(analysis):
    # Check if file is corrupted based on analysis results
    return (analysis['score'] == 0.0 and 
            analysis['features']['speech_rate'] == 0.0 and 
            analysis['features']['pause_freq'] == 0.0 and 
            len(analysis['errors']) == 0 and 
            not analysis['text'].strip())

def generate_predictions():
    # Generate predictions with resume capability
    print("Loading model and test data...")
    model = joblib.load(CONFIG['model_path'])
    test_df = pd.read_csv(CONFIG['test_csv'])
    checker = GrammarChecker()
    progress = load_progress()
    
    print(f"Resuming from {len(progress['processed'])} processed files")
    print(f"{len(test_df) - len(progress['processed'])} files remaining")
    
    # Prepare results DataFrame
    results_df = pd.DataFrame(progress['results'])
    
    # Process files
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        filename = row['filename']
        
        # Skip already processed files
        if filename in progress['processed']:
            continue
            
        audio_path = os.path.join(CONFIG['audio_dir'], filename)
        try:
            # Default low prediction if the file missing or corrupted
            if not os.path.exists(audio_path):
                pred = CONFIG['corrupted_score']
                print(f"Missing file detected: {filename} - assigning low score")
            else:
                analysis = checker.analyze_audio(audio_path)
                
                if is_corrupted_file(analysis):
                    pred = CONFIG['corrupted_score']
                    print(f"Corrupted file detected: {filename} - assigning low score")
                else:
                    features = np.array([[
                        analysis['score'],
                        analysis['features']['speech_rate'],
                        analysis['features']['pause_freq'],
                        len(analysis['errors'])
                    ]])
                    pred = max(0, min(5, model.predict(features)[0]))
            
            # Update results
            results_df = pd.concat([
                results_df,
                pd.DataFrame([{
                    'filename': filename,
                    'predicted_score': pred
                }])
            ], ignore_index=True)
            
            # Update progress
            progress['processed'].append(filename)
            progress['results'] = results_df.to_dict('records')
            
            # Periodic saving
            if idx % CONFIG['checkpoint_interval'] == 0:
                save_progress(progress)
                results_df.to_csv(CONFIG['output_file'], index=False)
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)} - assigning low score")
            results_df = pd.concat([
                results_df,
                pd.DataFrame([{
                    'filename': filename,
                    'predicted_score': CONFIG['corrupted_score']
                }])
            ], ignore_index=True)
    
    # Final save
    save_progress(progress)
    results_df.to_csv(CONFIG['output_file'], index=False)
    return results_df

def main():
    print("Starting prediction generation...")
    results = generate_predictions()
    
    print("\nPrediction Statistics:")
    print(results['predicted_score'].describe())
    print(f"\n Predictions saved to {CONFIG['output_file']}")

if __name__ == "__main__":
    main()