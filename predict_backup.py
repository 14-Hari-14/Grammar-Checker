def generate_test_predictions(model_path, test_csv_path, audio_dir):
    """
    Process test files and generate predictions for submission
    
    Args:
        model_path: Path to your trained model file
        test_csv_path: Path to test.csv with filenames
        audio_dir: Directory containing test audio files
    
    Returns:
        DataFrame with filename and predicted scores
    """
    import pandas as pd
    import numpy as np
    import os
    from tqdm import tqdm
    import joblib
    from utils import GrammarChecker
    
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    print(f"Reading test data from {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    print(f"Found {len(test_df)} test files to process")
    
    # Create results dataframe
    results_df = pd.DataFrame(columns=['filename', 'predicted_score'])
    
    # Process each test file
    checker = GrammarChecker()
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing test files"):
        filename = row['filename']
        audio_file = os.path.join(audio_dir, filename)
        
        try:
            # Check if file exists
            if not os.path.exists(audio_file):
                print(f"Warning: File not found {audio_file}")
                # Add default prediction
                results_df.loc[len(results_df)] = {
                    'filename': filename,
                    'predicted_score': 2.5  # Default middle score
                }
                continue
                
            # Analyze audio using same method as training
            analysis = checker.analyze_audio(audio_file)
            
            # Extract features in same format as training
            features = np.array([[
                analysis['score'],
                analysis['features']['speech_rate'],
                analysis['features']['pause_freq'],
                len(analysis['errors'])
            ]])
            
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Clip prediction to valid range (0-5)
            prediction = max(0, min(5, prediction))
            
            # Add to results
            results_df.loc[len(results_df)] = {
                'filename': filename,
                'predicted_score': prediction
            }
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            # Add default prediction
            results_df.loc[len(results_df)] = {
                'filename': filename,
                'predicted_score': 2.5  # Default middle score
            }
    
    print(f"Processed {len(results_df)} test files")
    
    # Optional: Show prediction statistics
    print("\nPrediction Statistics:")
    print(f"  Min score: {results_df['predicted_score'].min():.2f}")
    print(f"  Max score: {results_df['predicted_score'].max():.2f}")
    print(f"  Mean score: {results_df['predicted_score'].mean():.2f}")
    print(f"  Median score: {results_df['predicted_score'].median():.2f}")
    
    return results_df

def save_submission(results_df, output_path="submission.csv"):
    """Save results to submission file"""
    # Ensure the output has the right format
    submission_df = results_df[['filename', 'predicted_score']]
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    return output_path

def main():
    # Configuration
    MODEL_PATH = "grammar_model.pkl"  # Your trained model path
    TEST_CSV = "/home/hari/shl_dataset/test.csv"  # Path to test.csv
    TEST_AUDIO_DIR = "/home/hari/shl_dataset/audios/test"  # Directory with test audio files
    
    # Generate predictions
    results_df = generate_test_predictions(MODEL_PATH, TEST_CSV, TEST_AUDIO_DIR)
    
    # Save submission
    submission_path = save_submission(results_df)
    print(f"Done! Submission ready at {submission_path}")

if __name__ == "__main__":
    main()