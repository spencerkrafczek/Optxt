# gestures.py
import numpy as np
import joblib
import os

# Load gesture model (do this once at startup)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "training/models/gesture_model.pkl")

try:
    gesture_model = joblib.load(MODEL_PATH)
    print("✅ Gesture model loaded successfully")
except FileNotFoundError:
    gesture_model = None
    print("⚠️  Gesture model not found - train it first!")
    print(f"   Looking for: {MODEL_PATH}")

def extract_gesture_features(landmarks):
    """Convert landmarks to features for gesture model."""
    features = []
    
    # Shoulder positions (for shrug detection)
    if 'pose' in landmarks:
        pose = landmarks['pose']
        features.extend([pose['left_shoulder'][0], pose['left_shoulder'][1]])
        features.extend([pose['right_shoulder'][0], pose['right_shoulder'][1]])
    else:
        features.extend([0.5, 0.5, 0.5, 0.5])  # Default if no pose
    
    # Hand positions (for wave, middle finger)
    if 'hands' in landmarks and len(landmarks['hands']) > 0:
        hand = landmarks['hands'][0]
        features.extend([hand['wrist'][0], hand['wrist'][1]])
        features.extend([hand['middle_tip'][0], hand['middle_tip'][1]])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])  # No hand detected
    
    # Head position (for nod/shake)
    if 'face' in landmarks and 'nose_tip' in landmarks['face']:
        nose = landmarks['face']['nose_tip']
        chin = landmarks['face']['chin']
        features.extend([nose[0], nose[1], chin[0], chin[1]])
    else:
        features.extend([0.5, 0.5, 0.5, 0.5])
    
    return features

def detect_gesture(landmarks):
    """Predict gesture from landmarks."""
    if gesture_model is None:
        return "no_model"
    
    if landmarks is None:
        return "no_data"
    
    features = extract_gesture_features(landmarks)
    
    # Reshape for model (expects 2D array)
    features_array = np.array(features).reshape(1, -1)
    
    # Predict
    prediction = gesture_model.predict(features_array)[0]
    
    return prediction

# Test
if __name__ == "__main__":
    print("Testing gesture detection...")
    print(f"Model loaded: {gesture_model is not None}")
    
    # Dummy test
    dummy_landmarks = {
        'pose': {
            'left_shoulder': (0.3, 0.4, 0.0),
            'right_shoulder': (0.7, 0.4, 0.0)
        },
        'face': {
            'nose_tip': (0.5, 0.5, 0.0),
            'chin': (0.5, 0.7, 0.0)
        }
    }
    
    result = detect_gesture(dummy_landmarks)
    print(f"Test prediction: {result}")