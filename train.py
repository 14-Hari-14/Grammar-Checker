import os
import json
import warnings
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from utils import GrammarChecker

# Global warning suppression
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="whisper.*")

# Configuration
CONFIG = {
    "audio_paths": {
        "train": "/home/hari/shl_dataset/audios/train",
        "test": "/home/hari/shl_dataset/audios/test" 
    },
    "checkpoint": "progress.json",
    "model_path": "grammar_model.pkl",
    "max_files": 300,  # Safety limit for your deadline
    "min_file_size": 1024,  # 1KB minimum to avoid corrupt files
    "rf_params": {
        "n_estimators": 100,  # Reduced for speed
        "max_depth": 8,
        "n_jobs": -1,  # Use all cores
        "random_state": 42
    }
}

def validate_audio_file(path: str) -> bool:
    """Check if file exists and meets size requirements"""
    return os.path.exists(path) and os.path.getsize(path) >= CONFIG['min_file_size']

def load_checkpoint():
    """Load existing progress if available"""
    try:
        with open(CONFIG['checkpoint'], 'r') as f:
            data = json.load(f)
        print(f"Resuming from checkpoint ({len(data['processed_files'])} files done)")
        return set(data['processed_files']), data['features'], data['labels']
    except (FileNotFoundError, json.JSONDecodeError):
        return set(), [], []

def save_checkpoint(processed_files, features, labels):
    """Save current progress"""
    with open(CONFIG['checkpoint'], 'w') as f:
        json.dump({
            'processed_files': list(processed_files),
            'features': features,
            'labels': labels
        }, f)

def process_file(row):
    print(f"START Processing {row['filename']}")  # <-- ADD THIS LINE
    audio_path = os.path.join(CONFIG['audio_paths']['train'], row['filename'])
    
    if not validate_audio_file(audio_path):
        print(f"SKIPPED {row['filename']}")  # <-- ADD THIS
        return None
        
    try:
        checker = GrammarChecker()
        analysis = checker.analyze_audio(audio_path)
        print(f"COMPLETED {row['filename']}")  # <-- ADD THIS
        return (analysis['score'], analysis['features']['speech_rate'], 
                analysis['features']['pause_freq'], len(analysis['errors']),
                row['label'])
    except Exception as e:
        print(f"FAILED {row['filename']}: {str(e)}")  # <-- ADD THIS
        return None

# Update import statement


# In the main function, define tasks before using ProcessPoolExecutor
def main():
    # Load data
    train_df = pd.read_csv("/home/hari/shl_dataset/train.csv").head(CONFIG['max_files'])
    test_df = pd.read_csv("/home/hari/shl_dataset/test.csv")
    
    # Initialize progress
    processed_files, features, labels = load_checkpoint()
    
    # Define the tasks - these are the rows that need processing
    # Filter out already processed files
    tasks = [row for _, row in train_df.iterrows() 
             if row['filename'] not in processed_files]
    
    # Parallel processing
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_file, row): row for row in tasks}
        
        # This collects results as they complete
        for future in tqdm(as_completed(futures), total=len(tasks)):
            row = futures[future]
            result = future.result()
            if result:
                processed_files.add(row['filename'])
                features.append(result[:4])
                labels.append(result[4])
                save_checkpoint(processed_files, features, labels)
    
    # Remove the incorrect section using "results" variable
    # Instead, we already collected results in the loop above
    
    # Train model
    X_train, y_train = np.array(features), np.array(labels)
    model = RandomForestRegressor(**CONFIG['rf_params'])
    model.fit(X_train, y_train)
    
    # Save everything
    joblib.dump(model, CONFIG['model_path'])
    save_checkpoint(processed_files, features, labels)
    print(f"\n Done! Model saved to {CONFIG['model_path']}")
    print(f"Processed {len(features)}/{len(train_df)} files successfully")

if __name__ == '__main__':
    main()