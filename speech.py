import os
import platform
import subprocess
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import stream, save, VoiceSettings

load_dotenv()

# Setup caching directory
CACHE_DIR = "audio_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Initialize ElevenLabs
try:
    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    USE_ELEVEN = True
except Exception:
    USE_ELEVEN = False
    print("⚠️ ElevenLabs API Key not found. Using system TTS fallback.")

# PART 2: Dynamic Emotion Mapping
# Stability: Lower = more expressive. Style: Higher = more intense.
EMOTION_SETTINGS = {
    "happy": VoiceSettings(stability=0.4, similarity_boost=0.75, style=0.5),
    "angry": VoiceSettings(stability=0.2, similarity_boost=0.8, style=0.9),
    "sad": VoiceSettings(stability=0.3, similarity_boost=0.7, style=0.4),
    "neutral": VoiceSettings(stability=0.8, similarity_boost=0.75, style=0.0)
}

def say_interaction(text, emotion="neutral"):
    print(f"🔊 Optxt says: {text} ({emotion})")
    
    # PART 3: Phrase Caching
    clean_text = text.replace(" ", "_").lower()
    filepath = os.path.join(CACHE_DIR, f"{clean_text}_{emotion}.mp3")

    # If already cached, play locally (Zero Latency)
    if os.path.exists(filepath):
        play_local_audio(filepath)
        return

    # If not cached, use ElevenLabs
    if USE_ELEVEN:
        try:
            settings = EMOTION_SETTINGS.get(emotion, EMOTION_SETTINGS["neutral"])
            audio_stream = client.text_to_speech.stream(
                text=text,
                voice_id="JBFqnCBsd6RMkjVDRZzb", # Rachel
                model_id="eleven_flash_v2_5",    # Fastest model
                voice_settings=settings
            )
            
            # Stream for immediate playback
            stream(audio_stream)
            
            # Optional: Save for next time (Caching)
            # (Note: streaming and saving simultaneously requires a small buffer)
        except Exception as e:
            print(f"ElevenLabs Error: {e}")
            fallback_tts(text)
    else:
        fallback_tts(text)

def play_local_audio(path):
    """Play cached files using system tools"""
    if platform.system() == "Darwin":
        subprocess.run(["afplay", path])

def fallback_tts(text):
    if platform.system() == "Darwin":
        subprocess.run(["say", text])