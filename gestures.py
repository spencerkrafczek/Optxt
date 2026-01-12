# gestures.py
import numpy as np
import joblib
import os

# Load gesture model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "training/models/gesture_model.pkl")

try:
    gesture_model = joblib.load(MODEL_PATH)
    print("✅ Gesture model loaded successfully")
except FileNotFoundError:
    gesture_model = None
    print("⚠️  Gesture model not found - train it first!")

def extract_gesture_features(landmarks):
    """Convert landmarks to features (STATIC - NO MOTION)."""
    features = []
    
    # Shoulder positions (for shrug)
    if 'pose' in landmarks:
        pose = landmarks['pose']
        features.extend([pose['left_shoulder'][0], pose['left_shoulder'][1]])
        features.extend([pose['right_shoulder'][0], pose['right_shoulder'][1]])
        features.extend([pose['left_elbow'][0], pose['left_elbow'][1]])
        features.extend([pose['right_elbow'][0], pose['right_elbow'][1]])
    else:
        features.extend([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    
    # Hand positions (for thumbs, pointing, middle finger)
    if 'hands' in landmarks and len(landmarks['hands']) > 0:
        hand = landmarks['hands'][0]
        features.extend([hand['wrist'][0], hand['wrist'][1]])
        features.extend([hand['thumb_tip'][0], hand['thumb_tip'][1]])
        features.extend([hand['index_tip'][0], hand['index_tip'][1]])
        features.extend([hand['middle_tip'][0], hand['middle_tip'][1]])
        features.extend([hand['ring_tip'][0], hand['ring_tip'][1]])
        features.extend([hand['pinky_tip'][0], hand['pinky_tip'][1]])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    return features

def detect_gesture(landmarks):
    """Predict gesture from landmarks (SINGLE FRAME)."""
    if gesture_model is None:
        return "no_model"
    
    if landmarks is None:
        return "no_data"
    
    features = extract_gesture_features(landmarks)
    features_array = np.array(features).reshape(1, -1)
    
    # Predict
    prediction = gesture_model.predict(features_array)[0]
    
    return prediction

# Test
if __name__ == "__main__":
    print("Testing gesture detection (static poses)...")
    print(f"Model loaded: {gesture_model is not None}")