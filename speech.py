import pyttsx3

engine = pyttsx3.init()

def say_interaction(text):
    print(f"Optxt says: {text}")
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    say_interaction("happy happy happy happy happy")