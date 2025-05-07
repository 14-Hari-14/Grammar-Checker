# Spoken Grammar Assessment System

This project implements an automated system that evaluates the grammatical quality of spoken language in audio recordings. It analyzes audio files and predicts Mean Opinion Score (MOS) grammar scores on a scale from 0 to 5 — 0 being the lowest and 5 being the best.

## Project Overview

The system processes audio recordings of 45-60 seconds in length, transcribes the speech, analyzes various linguistic features, and predicts grammar quality scores that correlate with human expert assessments.

### Dataset

- **Training dataset**: 444 audio samples with expert-assigned MOS scores
- **Testing dataset**: 195 audio samples for evaluation
- **Audio format**: 45–60 second spoken recordings

## Technical Approach

### 1. Audio Processing Pipeline

The system implements a multi-stage pipeline:

1. **Speech-to-Text Conversion**: Uses OpenAI's Whisper model to transcribe spoken audio to text
2. **Grammar Feature Extraction**: Analyzes transcribed text using LanguageTool to identify grammar errors
3. **Feature Engineering**: Extracts features including:

   - Grammar error counts (categorized by severity)
   - Speech rate metrics
   - Pause frequency
   - Text complexity measures

4. **Prediction Model**: Uses a Random Forest Regressor to predict grammar quality scores
5. **Evaluation**: Measures model performance using Mean Absolute Error and R²

### 2. Implementation Details

#### Key Components

- **GrammarChecker**: Core class that handles transcription and grammar analysis
- **Training Pipeline**: Processes audio files in parallel while also creating checkpoints to store progress
- **Prediction System**: Generates grammar scores for new audio files

#### Technologies Used

- **Speech Recognition**: OpenAI Whisper
- **Grammar Analysis**: LanguageTool (Python wrapper)
- **Audio Processing**: Librosa, PyDub
- **Machine Learning**: Scikit-learn (RandomForestRegressor)
- **Data Processing**: Pandas, NumPy
- **Parallel Processing**: Python's `concurrent.futures`

## Results

The model achieved:

- **Mean Absolute Error**: 0.48–0.52 on validation data
  - (The average of the absolute differences between predicted and actual values, indicating overall prediction error.)
- **R² Score**: 0.65–0.72 on validation data
  - (Measures how well the model explains the variability of the target; closer to 1 means better fit.)
- Distribution of predicted scores closely matched the training distribution

## Implemented Optimizations

While working on this project I made several improvements to boost performance and reliability:

### Whisper Integration

- **Initial:** Used OpenAI's Whisper API (required network, had rate limits)
- **Optimized:** Switched to local Whisper model (runs offline)
- **Impact:** 4× faster processing, no API limits or costs

### Compute Efficiency

- **Initial:** CPU-only processing
- **Optimized:** Added GPU support for Whisper and feature extraction
- **Impact:** 8–12× faster transcription

### Progress Tracking

- **Initial:** Required full reprocessing if interrupted
- **Optimized:** Added checkpoints for:

  - Processed file list
  - Extracted features
  - Partial model state

- **Impact:** Saved 50+ hours of reprocessing

### Corrupted File Handling

- **Initial:** Corrupted audio caused silent prediction errors
- **Optimized:** Added multi-stage validation:

  - File size checks
  - Audio amplitude analysis
  - Zero-feature detection

- **Impact:** Catching about 90% of all corrupted files

### Parallel Processing

- **Initial:** Sequential file processing
- **Optimized:** Used `concurrent.futures` with 4–8 workers
- **Impact:** 3–5× faster feature extraction

## Future Optimization Ideas

### Feature Engineering

- Add more linguistic metrics: syntactic complexity, lexical diversity
- Incorporate speaker-specific factors for better generalization

### Model Improvements

- Use deep learning models like CNNs/RNNs for direct audio input
- Leverage language models like BERT for better text analysis
- Try ensemble models to combine strengths of different approaches

### Performance Optimization

- Add support for distributed computing
- Extend GPU acceleration beyond Whisper
- Improve caching to avoid repeated processing

### Evaluation Enhancements

- Include feedback loop with human evaluations
- Add more rigorous cross-validation techniques
- Include confidence scores for each prediction

## Usage Instructions

### Setup

```bash
# Clone repository
git clone https://github.com/username/grammar-assessment-system.git
cd grammar-assessment-system

# Install dependencies
pip install -r requirements.txt

# Install local whisper
pip install git+https://github.com/openai/whisper.git
```

### Training

```bash
python train.py --train_csv path/to/train.csv --audio_dir path/to/audio/files
```

### Prediction

```bash
python predict.py --test_csv path/to/test.csv --audio_dir path/to/test/audio
```

## Project Structure

```
grammar-assessment-system/
├── utils.py           # Core grammar checking utilities
├── train.py           # Model training pipeline
├── predict.py         # Prediction generation for test data
├── requirements.txt   # Project dependencies
└── README.md          # This documentation
```

## Limitations and Considerations

- Only optimized for English audio inputs
- Background noise or unclear speech can affect accuracy
- Heavy computation — requires decent hardware
- Model outputs might still miss subtle grammar issues
- Some corrupted files might occasionally get through despite checks

## Future Work

1. Support for multiple languages
2. Optimize for low-latency / real-time use
3. Add feedback-based grammar suggestions

## Requirements

- Python 3.8+
- OpenAI API key (or use local Whisper)
- 8GB+ RAM recommended
- See requirements.txt for full list

## License

MIT License

## Output Examples

Prediction output also shows which files were skipped due to corruption:

![prediction output](image-1.png)

Training sometimes had issues when multiple processes clashed. This screenshot shows one such error (checkpoints were created before retrying):

![errors faced during training](shl_error_2.png)

## Acknowledgments

- OpenAI for the Whisper model
- LanguageTool for grammar analysis tools
- PyTorch, NumPy, and scikit-learn for model training and data handling
