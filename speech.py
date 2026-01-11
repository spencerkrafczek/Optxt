# speech.py
import os
import platform
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup caching directory
CACHE_DIR = "audio_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Try to initialize ElevenLabs
USE_ELEVEN = False
try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import save, VoiceSettings
    
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if api_key:
        client = ElevenLabs(api_key=api_key)
        USE_ELEVEN = True
        print("✅ ElevenLabs TTS enabled")
    else:
        print("⚠️  ELEVENLABS_API_KEY not found in .env file")
except ImportError:
    print("⚠️  elevenlabs package not installed. Run: pip install elevenlabs")
except Exception as e:
    print(f"⚠️  ElevenLabs setup failed: {e}")

if not USE_ELEVEN:
    print("   Using system TTS fallback (Mac 'say' command)")

# Emotion-based voice settings
EMOTION_SETTINGS = {
    "happy": VoiceSettings(stability=0.4, similarity_boost=0.75, style=0.5),
    "angry": VoiceSettings(stability=0.2, similarity_boost=0.8, style=0.9),
    "shocked": VoiceSettings(stability=0.3, similarity_boost=0.8, style=0.7),
    "neutral": VoiceSettings(stability=0.8, similarity_boost=0.75, style=0.0)
}

def say_interaction(text, emotion="neutral"):
    """
    Speak text with emotion-based voice settings.
    Uses ElevenLabs if available, falls back to system TTS.
    """
    print(f"🔊 [{emotion}] {text}")
    
    # Check cache first
    clean_text = text.replace(" ", "_").replace(",", "").replace(".", "").lower()[:50]
    filepath = os.path.join(CACHE_DIR, f"{clean_text}_{emotion}.mp3")
    
    if os.path.exists(filepath):
        # Play cached audio
        play_local_audio(filepath)
        return
    
    # Generate new audio
    if USE_ELEVEN:
        try:
            # Generate audio with emotion settings
            audio = client.text_to_speech.convert(
                text=text,
                voice_id="EXAVITQu4vr4xnSDxMaL",  # George voice (calm, clear)
                model_id="eleven_flash_v2_5",  # Fast model
                voice_settings=EMOTION_SETTINGS.get(emotion, EMOTION_SETTINGS["neutral"])
            )
            
            # Save to cache
            save(audio, filepath)
            
            # Play
            play_local_audio(filepath)
            
        except Exception as e:
            print(f"ElevenLabs error: {e}")
            fallback_tts(text)
    else:
        # Use system TTS
        fallback_tts(text)

def play_local_audio(path):
    """Play cached audio files using system tools."""
    if platform.system() == "Darwin":  # Mac
        subprocess.run(["afplay", path], check=False)
    elif platform.system() == "Windows":
        # Windows Media Player
        subprocess.run(["start", path], shell=True, check=False)
    else:  # Linux
        subprocess.run(["mpg123", path], check=False)

def fallback_tts(text):
    """System TTS fallback."""
    if platform.system() == "Darwin":  # Mac
        subprocess.run(["say", text])
    elif platform.system() == "Windows":
        # Windows SAPI
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

# Test
if __name__ == "__main__":
    print("Testing TTS system...")
    
    say_interaction("Hello! Testing neutral voice.", emotion="neutral")
    say_interaction("I'm so happy to see you!", emotion="happy")
    say_interaction("I am very angry right now!", emotion="angry")
    say_interaction("Oh my god, that's shocking!", emotion="shocked")
    
    print("\n✅ TTS test complete!")