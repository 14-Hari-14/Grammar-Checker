import os
from utils import GrammarChecker
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

def analyze_problem_file(audio_path):
    """Deep analysis of a single problematic audio file"""
    print(f"\n🔍 Analyzing {os.path.basename(audio_path)}")
    
    # 1. Basic file validation
    try:
        file_size = os.path.getsize(audio_path)
        print(f"File size: {file_size/1024:.2f} KB")
        if file_size < 1024:
            print("⚠️ File is too small (likely corrupt)")
    except Exception as e:
        print(f"File error: {str(e)}")
        return

    # # 2. Audio loading check
    # try:
    #     y, sr = sf.read(audio_path)
    #     print(f"Audio loaded - Duration: {len(y)/sr:.2f}s, Sample rate: {sr}Hz")
        
    #     # Visualize waveform
    #     plt.figure(figsize=(12, 4))
    #     plt.plot(y[:sr*3])  # First 3 seconds
    #     plt.title(f"Waveform: {os.path.basename(audio_path)}")
    #     plt.show()
        
    # except Exception as e:
    #     print(f"Failed to load with soundfile: {str(e)}")
    #     try:
    #         y, sr = librosa.load(audio_path, sr=None)
    #         print(f"Loaded with librosa - Duration: {len(y)/sr:.2f}s")
    #     except Exception as le:
    #         print(f"Completely unreadable: {str(le)}")
    #         return

    # 3. Grammar analysis
    checker = GrammarChecker()
    try:
        analysis = checker.analyze_audio(audio_path)
        print("\n📊 Analysis Results:")
        print(f"Grammar Score: {analysis['score']:.2f}/5")
        print(f"Speech Rate: {analysis['features']['speech_rate']:.2f} words/sec")
        print(f"Pause Frequency: {analysis['features']['pause_freq']:.2f}/sec")
        print(f"Error Count: {len(analysis['errors'])}")
        
        print("\n❌ Detected Errors:")
        for error in analysis['errors'][:5]:  # Show first 5 errors
            print(f"- {error['type']}: {error['message']}")
            
    except Exception as e:
        print(f"Analysis failed: {str(e)}")

    # 4. Raw transcription
    try:
        transcription = checker.transcribe_audio(audio_path)
        print("\n📝 Transcription:")
        print(transcription[:500] + "...")  # First 500 chars
    except Exception as e:
        print(f"Transcription failed: {str(e)}")

# Example usage
problem_files = [
    "/home/hari/shl_dataset/audios/train/audio_1156.wav",
    "/home/hari/shl_dataset/audios/train/audio_845.wav",
    "/home/hari/shl_dataset/audios/test/audio_415.wav"
    # Add other problematic files
]

for file_path in problem_files:
    analyze_problem_file(file_path)
    print("\n" + "="*80 + "\n")