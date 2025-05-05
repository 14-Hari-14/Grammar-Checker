import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from utils import GrammarChecker
import json

# Initialize feature extractor
checker = GrammarChecker()

# Define paths - UPDATE THESE TO MATCH YOUR SYSTEM
BASE_AUDIO_PATH = "/home/hari/shl_dataset/audios"
TRAIN_CSV_PATH = "/home/hari/shl_dataset/train.csv"
TEST_CSV_PATH = "/home/hari/shl_dataset/test.csv"

# 1. Load training data
train_df = pd.read_csv(TRAIN_CSV_PATH)

# 2. Extract features and labels from training set
train_features = []
train_labels = []

for idx, row in train_df.iterrows():
    audio_path = os.path.join(BASE_AUDIO_PATH, "train", row['filename'])
    
    try:
        analysis = checker.analyze_audio(audio_path)
        
        # Skip if feature extraction failed
        if 'features' not in analysis:
            print(f"Skipping {row['filename']} - feature extraction failed")
            continue
            
        train_features.append([
            analysis['score'],
            analysis['features']['speech_rate'],
            analysis['features']['pause_freq'],
            len(analysis['errors'])
        ])
        train_labels.append(row['label'])
        
    except Exception as e:
        print(f"Error processing {row['filename']}: {str(e)}")
        continue

# 3. Load test data
test_df = pd.read_csv(TEST_CSV_PATH)

# 4. Extract features and labels from test set
test_features = []
test_labels = []

for idx, row in test_df.iterrows():
    audio_path = os.path.join(BASE_AUDIO_PATH, "test", row['filename'])
    
    try:
        analysis = checker.analyze_audio(audio_path)
        
        if 'features' not in analysis:
            print(f"Skipping {row['filename']} - feature extraction failed")
            continue
            
        test_features.append([
            analysis['score'],
            analysis['features']['speech_rate'],
            analysis['features']['pause_freq'],
            len(analysis['errors'])
        ])
        test_labels.append(row['label'])
        
    except Exception as e:
        print(f"Error processing {row['filename']}: {str(e)}")
        continue
    
with open('progress_checkpoint.json', 'w') as f:
    json.dump({
        'processed_files': [row['filename'] for row in train_df.iterrows() 
                          if os.path.join(BASE_AUDIO_PATH, "train", row['filename']) in processed_audios],
        'train_features': train_features,
        'train_labels': train_labels
    }, f)

# Rest of your code remains the same...
X_train = np.array(train_features)
y_train = np.array(train_labels)
X_test = np.array(test_features)
y_test = np.array(test_labels)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
print(f"Test MAE: {mean_absolute_error(y_test, test_pred):.2f}")
print(f"Test R²: {r2_score(y_test, test_pred):.2f}")

joblib.dump(model, "grammar_model.pkl")
print("Model saved to grammar_model.pkl")


import os
import json
import warnings
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from utils import GrammarChecker

# Global warning suppression
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="whisper.*")

CONFIG = {
    "audio_paths": {
        "train": "/home/hari/shl_dataset/audios/train",
        "test": "/home/hari/shl_dataset/audios/test" 
    },
    "checkpoint": "progress.json",
    "model_path": "grammar_model.pkl",
    "rf_params": {
        "n_estimators": 200,
        "max_depth": 10,
        "n_jobs": -1,
        "random_state": 42
    }
}

def load_checkpoint():
    """Load existing progress if available"""
    try:
        with open(CONFIG['checkpoint'], 'r') as f:
            data = json.load(f)
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

def init_worker():
    """Initialize each parallel worker"""
    warnings.filterwarnings("ignore")
    global checker
    checker = GrammarChecker()

def process_file(args):
    """Process single file with checkpoint awareness"""
    row, processed_files = args
    if row['filename'] in processed_files:
        return None
        
    try:
        audio_path = os.path.join(CONFIG['audio_paths']['train'], row['filename'])
        analysis = checker.analyze_audio(audio_path)
        return (
            analysis['score'],
            analysis['features']['speech_rate'],
            analysis['features']['pause_freq'],
            len(analysis['errors']),
            row['label']
        )
    except Exception as e:
        print(f"Error processing {row['filename']}: {str(e)}")
        return None

def train_model(features, labels):
    """Train and evaluate the model"""
    X_train = np.array([x[:4] for x in features])
    y_train = np.array(labels)
    
    model = RandomForestRegressor(**CONFIG['rf_params'])
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, CONFIG['model_path'])
    print(f"Model saved to {CONFIG['model_path']}")
    
    return model

def main():
    # Load data and checkpoint
    train_df = pd.read_csv("/home/hari/shl_dataset/train.csv")
    test_df = pd.read_csv("/home/hari/shl_dataset/test.csv")
    processed_files, features, labels = load_checkpoint()
    
    # Parallel processing
    with ProcessPoolExecutor(
        initializer=init_worker,
        max_workers=multiprocessing.cpu_count()
    ) as executor:
        tasks = [(row, processed_files) for _, row in train_df.iterrows()]
        results = list(tqdm(
            executor.map(process_file, tasks),
            total=len(tasks),
            desc="Processing Files"
        ))
    
    # Update training data
    new_results = [r for r in results if r is not None]
    features.extend([r[:4] for r in new_results])
    labels.extend([r[4] for r in new_results])
    processed_files.update(row['filename'] for _, row in train_df.iterrows() 
                         if row['filename'] not in processed_files)
    
    # Save progress
    save_checkpoint(processed_files, features, labels)
    
    # Train and save model
    train_model(features, labels)

if __name__ == '__main__':
    main()