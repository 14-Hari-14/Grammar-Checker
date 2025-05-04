import openai #For whisper
import language_tool_python #To check grammar mistakes 
import librosa  #package to analayze audio
from pydub import AudioSegment #manipulate audio
import os  #os dependant functionality
from typing import Union
AUDIO_DIR = "../shl_dataset/audios/" #Directory where audio files are stored 


class GrammarChecker:
    def __init__(self):
        self.tool = language_tool_python.LanguageTool('en-US', config={ 'cacheSize': 1000, 'pipelineCaching': True })
        
        self.ERRORS = {
            # Verb Errors
            'EN_VERB_AGREEMENT',       # "He go"
            'EN_VERB_TENSE',           # "I seen it"
            'EN_INFINITIVE',           # "She can to go"
            'EN_PAST_PARTICIPLE',      # "I have ate"
            'EN_IRREGULAR_VERBS',      # "She swimmed"
            
            # Pronoun Errors
            'EN_PRONOMINAL_AGREEMENT', # "Me and him went"
            'EN_CASE',                 # "Between you and I"
            
            # Article Errors
            'EN_A_VS_AN',              # "a apple"
            'EN_ARTICLE_REQUIRED',     # "She is doctor"
            
            # Preposition Errors
            'EN_PREPOSITIONS',         # "Dependent of"
            'EN_CONFUSED_PREPOSITION', # "On the picture" (should be "in")
            
            # Word Order
            'EN_WORD_ORDER',           # "I yesterday went"
            'EN_QUANTIFIER_ORDER',     # "I have much books"
            
            # Negation
            'EN_DOUBLE_NEGATIVE',      # "I don't know nothing"
            'EN_NEGATION_MISUSE',      # "I no like it"
            
            # Question Formation
            'EN_QUESTION_FORM',        # "You are coming?"
            'EN_QUESTION_TAGS',        # "You like it, is it?"
            
            # Count/Non-count Nouns
            'EN_COUNTABILITY',         # "many informations"
            'EN_PLURAL_SINGULAR',      # "three child"
            
            # Modals
            'EN_MODAL_VERBS',          # "She must to go"
            'EN_MODAL_VERB_SEQUENCE',  # "He can goes"
            
            # Conditionals
            'EN_CONDITIONALS',         # "If I will go"
            
            # Reported Speech
            'EN_REPORTED_SPEECH',      # "She said she is coming"
            
            # Comparative/Superlative
            'EN_COMPARATIVES',         # "more better"
            'EN_SUPERLATIVES',         # "the most happy"
            
            # Contractions
            'EN_CONTRACTIONS',         # "I am" → "I'm" (formal vs informal)
            
            # Fillers
            'EN_FILLERS',              # "um", "uh" (fluency markers)
            
            # Informal Usage
            'EN_INFORMAL_WORDS',       # "gonna", "wanna"
            'EN_SLANG',                # "ain't", "y'all"
            
            # Pronunciation-based
            'EN_HOMOPHONE_ERRORS',     # "their/there", "your/you're"
            
            # # Plain english rule
            # 'EN_PLAIN_ENGLISH',  
            
            # # Repetition
            # 'EN_REPEATED_WORDS',       # "the the"
        }
    
    def transcribe_audio(self, audio_filename:str) -> str:
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        with open(audio_path, "rb") as audio_file:
            result = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            ).strip() 
        return result
    
    def _get_error_weight(self, rule_id: str) -> float:
        # Very bad mistakes 
        if rule_id in {
            'EN_VERB_AGREEMENT', 
            'EN_VERB_TENSE',
            'EN_PRONOMINAL_AGREEMENT',
            'EN_CASE'
        }:
            return 3.0
        
        # Pretty bad mistakes
        elif rule_id in {
            'EN_PREPOSITIONS',
            'EN_ARTICLE_REQUIRED',
            'EN_PAST_PARTICIPLE',
            'EN_QUESTION_FORM'
        }:
            return 2.0
        
        # Not too bad 
        else:
            return 1.0  # All other errors
    
    def extract_audio_features(self, audio_filename: str) -> dict[str, float]:
        #Extract speech rate and pause frequency
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        y, sr = librosa.load(audio_path)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Speech rate (words/sec)
        text = self.transcribe_audio(audio_filename)
        words = len(text.split())
        
        # Pause frequency (pauses/sec)
        intervals = librosa.effects.split(y, top_db=20)  # Detect silent pauses
        
        return {
            "speech_rate": words / duration if duration > 0 else 0,
            "pause_freq": len(intervals) / duration if duration > 0 else 0
        }
    
    def analyze_audio(self, audio_filename: str) -> dict[str, Union[float, list[dict]]]:
        text = self.transcribe_audio(audio_filename)
        if not text.strip():
            return {"score": 0.0, "errors": []}
        
        # Grammar analysis
        matches = self.tool.check(text)
        errors = []
        total_weight = 0.0
        
        for mistake in matches:
            if mistake.ruleId in self.ERRORS:
                weight = self._get_error_weight(mistake.ruleId)
                errors.append({
                    "type": mistake.ruleId,
                    "message": mistake.message,
                    "correction": mistake.replacements[0] if mistake.replacements else "",
                    "weight": weight
                })
                total_weight += weight
        
        # Calculate score (0-5 scale)
        words = len(text.split())
        score = max(0.0, min(5.0, 5 - (total_weight / max(1, words)) * 1.5))
        
        return {
            "score": round(score, 2),
            "errors": errors,
            "text": text
        }
    
    def safe_transcribe(self, audio_filename: str) -> str:
        try:
            return self.transcribe_audio(audio_filename)
        except Exception as e:
            print(f"Error transcribing {audio_filename}: {str(e)}")
            return ""

    
        

        