# emotions.py
import numpy as np
import joblib
import os

# Load emotion model (do this once at startup)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "training/models/emotion_model.pkl")

try:
    emotion_model = joblib.load(MODEL_PATH)
    print("✅ Emotion model loaded successfully")
except FileNotFoundError:
    emotion_model = None
    print("⚠️  Emotion model not found - train it first!")
    print(f"   Looking for: {MODEL_PATH}")

def extract_emotion_features(landmarks):
    """Convert face landmarks to features for emotion model."""
    features = []
    
    if 'face' not in landmarks:
        return None
    
    face = landmarks['face']
    
    # Flatten all face coordinates in sorted order (x, y only)
    for key in sorted(face.keys()):
        features.extend([face[key][0], face[key][1]])
    
    return features

def detect_emotion(landmarks):
    """Predict emotion from landmarks."""
    if emotion_model is None:
        return "no_model"
    
    if landmarks is None:
        return "no_data"
    
    features = extract_emotion_features(landmarks)
    
    if features is None:
        return "no_face"
    
    # Reshape for model (expects 2D array)
    features_array = np.array(features).reshape(1, -1)
    
    # Predict
    prediction = emotion_model.predict(features_array)[0]
    
    return prediction

# Test
if __name__ == "__main__":
    print("Testing emotion detection...")
    print(f"Model loaded: {emotion_model is not None}")
    
    # Dummy test
    dummy_landmarks = {
        'face': {
            'mouth_left': (0.3, 0.6, 0.0),
            'mouth_right': (0.7, 0.6, 0.0),
            'nose_tip': (0.5, 0.5, 0.0),
            'chin': (0.5, 0.8, 0.0),
            'forehead': (0.5, 0.2, 0.0)
        }
    }
    
    result = detect_emotion(dummy_landmarks)
    print(f"Test prediction: {result}")