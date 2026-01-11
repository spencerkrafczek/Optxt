import pyttsx3
import subprocess
import platform

# Try pyttsx3 first
try:
    engine = pyttsx3.init()
    USE_PYTTSX3 = True
except:
    USE_PYTTSX3 = False
    print("⚠️  pyttsx3 failed, using system TTS")

def say_interaction(text):
    print(f"🔊 Optxt says: {text}")
    
    if USE_PYTTSX3:
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")
            # Fallback to Mac system voice
            if platform.system() == "Darwin":  # Mac
                subprocess.run(["say", text])
    else:
        # Use Mac system voice directly
        if platform.system() == "Darwin":
            subprocess.run(["say", text])

# Test
if __name__ == "__main__":
    say_interaction("Voice system is online.")