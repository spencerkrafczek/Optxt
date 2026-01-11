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

# Updated snippet for speech.py
def say_interaction(text, emotion="neutral"):
    clean_text = text.replace(" ", "_").lower()
    filepath = os.path.join(CACHE_DIR, f"{clean_text}_{emotion}.mp3")

    if os.path.exists(filepath):
        # afplay is built into macOS and is very reliable for local playback
        subprocess.run(["afplay", filepath])
        return

    if USE_ELEVEN:
        try:
            # We use .convert() here to get the full audio bytes for saving
            audio = client.text_to_speech.convert(
                text=text,
                voice_id="JBFqnCBsd6RMkjVDRZzb",
                model_id="eleven_flash_v2_5",
                voice_settings=EMOTION_SETTINGS.get(emotion)
            )
            # Save the file to your cache folder
            save(audio, filepath)
            play_local_audio(filepath)
        except Exception as e:
            fallback_tts(text)

def play_local_audio(path):
    """Play cached files using system tools"""
    if platform.system() == "Darwin":
        subprocess.run(["afplay", path])

def fallback_tts(text):
    if platform.system() == "Darwin":
        subprocess.run(["say", text])