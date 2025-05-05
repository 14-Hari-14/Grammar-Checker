import os
from typing import Dict, List, Union
import language_tool_python
import librosa
import soundfile as sf
import whisper
import warnings


# Suppress all Whisper warnings
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")

class GrammarChecker:
    def __init__(self):
        # Initialize with silent Whisper model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.whisper_model = whisper.load_model("base").float()  # Force FP32
            self.tool = language_tool_python.LanguageTool('en-US')
        
        self.ERRORS = {
            'EN_VERB_AGREEMENT': 3.0,
            'EN_VERB_TENSE': 3.0,
            'EN_PREPOSITIONS': 2.0,
            'EN_ARTICLE_REQUIRED': 2.0,
            'EN_QUESTION_FORM': 2.0,
            'EN_INFINITIVE': 1.0,
            'EN_FILLERS': 0.5
        }

    def transcribe_audio(self, audio_path: str) -> str:
        # There were some corrupted files in the dataset so added a try catch block
        try:
            # First validate audio file
            if not os.path.exists(audio_path):
                raise ValueError(f"File not found: {audio_path}")
                
            # Check file size (empty files cause segfaults)
            if os.path.getsize(audio_path) < 1024:  # Less than 1KB
                raise ValueError("File too small (likely corrupt)")
                
            # Convert to WAV if needed (Whisper works best with WAV)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    result = self.whisper_model.transcribe(audio_path)
                    return result["text"].strip()
                except RuntimeError as e:
                    # Fallback to librosa loading
                    y, sr = librosa.load(audio_path, sr=16000)  # Force 16kHz
                    temp_path = "/tmp/audio_temp.wav"
                    librosa.output.write_wav(temp_path, y, sr)
                    result = self.whisper_model.transcribe(temp_path)
                    os.remove(temp_path)
                    return result["text"].strip()
                    
        except Exception as e:
            print(f"CRITICAL: Failed to process {audio_path}: {str(e)}")
            return ""

    def load_audio(self, audio_path: str):
        """Optimized audio loading with fallbacks"""
        try:
            return sf.read(audio_path)
        except Exception as e:
            print(f"SoundFile failed, using librosa: {str(e)}")
            return librosa.load(audio_path)

    def analyze_audio(self, audio_filename: str) -> Dict[str, Union[float, List[Dict], Dict]]:
        """Silent analysis with comprehensive error handling"""
        default_result = {
            'score': 0.0, 
            'errors': [], 
            'features': {'speech_rate': 0.0, 'pause_freq': 0.0},
            'text': ''
        }
        
        try:
            audio_path = os.path.join("/home/hari/shl_dataset/audios/train", audio_filename)
            y, sr = self.load_audio(audio_path)
            duration = librosa.get_duration(y=y, sr=sr)
            
            text = self.transcribe_audio(audio_path)
            if not text.strip():
                return default_result
                
            matches = self.tool.check(text)
            errors = []
            total_weight = 0.0
            
            for mistake in matches:
                if mistake.ruleId in self.ERRORS:
                    weight = self.ERRORS[mistake.ruleId]
                    errors.append({
                        'type': mistake.ruleId,
                        'message': mistake.message,
                        'weight': weight
                    })
                    total_weight += weight
            
            words = len(text.split())
            score = max(0.0, min(5.0, 5 - (total_weight / max(1, words) * 1.5)))
            
            intervals = librosa.effects.split(y, top_db=20)
            
            return {
                'score': round(score, 2),
                'errors': errors,
                'features': {
                    'speech_rate': words / duration if duration > 0 else 0,
                    'pause_freq': len(intervals) / duration if duration > 0 else 0
                },
                'text': text
            }
            
        except Exception as e:
            print(f"Analysis failed for {audio_filename}: {str(e)}")
            return default_result