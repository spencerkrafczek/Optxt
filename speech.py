import pyttsx3

engine = pyttsx3.init()

def say_interaction(text):
    print(f"SocialSight says: {text}")
    engine.say(text)
    engine.runAndWait()

# Test it (this only runs if you run this file directly)
if __name__ == "__main__":
    say_interaction("Voice system is online.")