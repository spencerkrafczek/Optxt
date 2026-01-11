# gestures.py
import numpy as np
import joblib
import os
from collections import deque

# Load gesture model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "training/models/gesture_model.pkl")

try:
    gesture_model = joblib.load(MODEL_PATH)
    print("✅ Gesture model loaded successfully")
except FileNotFoundError:
    gesture_model = None
    print("⚠️  Gesture model not found - train it first!")

# Store last 5 frames for temporal features (motion tracking)
landmark_history = deque(maxlen=5)

def extract_gesture_features(landmarks_sequence):
    """Convert landmarks sequence to features INCLUDING temporal motion."""
    
    # Use the LAST frame for spatial features
    landmarks = landmarks_sequence[-1]
    features = []
    
    # === SPATIAL FEATURES (current pose) ===
    
    # Shoulder positions
    if 'pose' in landmarks:
        pose = landmarks['pose']
        features.extend([pose['left_shoulder'][0], pose['left_shoulder'][1]])
        features.extend([pose['right_shoulder'][0], pose['right_shoulder'][1]])
        features.extend([pose['left_elbow'][0], pose['left_elbow'][1]])
        features.extend([pose['right_elbow'][0], pose['right_elbow'][1]])
    else:
        features.extend([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    
    # Hand positions
    if 'hands' in landmarks and len(landmarks['hands']) > 0:
        hand = landmarks['hands'][0]
        features.extend([hand['wrist'][0], hand['wrist'][1]])
        features.extend([hand['thumb_tip'][0], hand['thumb_tip'][1]])
        features.extend([hand['index_tip'][0], hand['index_tip'][1]])
        features.extend([hand['middle_tip'][0], hand['middle_tip'][1]])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Head position
    if 'face' in landmarks and 'nose_tip' in landmarks['face']:
        nose = landmarks['face']['nose_tip']
        chin = landmarks['face']['chin']
        features.extend([nose[0], nose[1], chin[0], chin[1]])
    else:
        features.extend([0.5, 0.5, 0.5, 0.5])
    
    # === TEMPORAL FEATURES (motion over time) ===
    
    if len(landmarks_sequence) >= 5:
        old_landmarks = landmarks_sequence[0]
        new_landmarks = landmarks_sequence[-1]
        
        # Head motion (nod/shake)
        if ('face' in old_landmarks and 'nose_tip' in old_landmarks['face'] and
            'face' in new_landmarks and 'nose_tip' in new_landmarks['face']):
            
            old_nose = old_landmarks['face']['nose_tip']
            new_nose = new_landmarks['face']['nose_tip']
            
            motion_x = new_nose[0] - old_nose[0]  # Left/right (shake)
            motion_y = new_nose[1] - old_nose[1]  # Up/down (nod)
            
            features.extend([motion_x, motion_y])
        else:
            features.extend([0.0, 0.0])
        
        # Shoulder motion (shrug)
        if ('pose' in old_landmarks and 'pose' in new_landmarks):
            old_left_sh_y = old_landmarks['pose']['left_shoulder'][1]
            new_left_sh_y = new_landmarks['pose']['left_shoulder'][1]
            shoulder_motion = new_left_sh_y - old_left_sh_y
            features.append(shoulder_motion)
        else:
            features.append(0.0)
        
        # Hand motion (wave)
        if ('hands' in old_landmarks and len(old_landmarks['hands']) > 0 and
            'hands' in new_landmarks and len(new_landmarks['hands']) > 0):
            
            old_wrist_x = old_landmarks['hands'][0]['wrist'][0]
            new_wrist_x = new_landmarks['hands'][0]['wrist'][0]
            hand_motion_x = new_wrist_x - old_wrist_x
            features.append(hand_motion_x)
        else:
            features.append(0.0)
    else:
        # Not enough frames yet
        features.extend([0.0, 0.0, 0.0, 0.0])
    
    return features

def detect_gesture(landmarks):
    """Predict gesture from landmarks with temporal tracking."""
    if gesture_model is None:
        return "no_model"
    
    if landmarks is None:
        return "no_data"
    
    # Add current frame to history
    landmark_history.append(landmarks)
    
    # Need at least 5 frames for motion tracking
    if len(landmark_history) < 5:
        return "warming_up"
    
    # Extract features from sequence
    features = extract_gesture_features(list(landmark_history))
    features_array = np.array(features).reshape(1, -1)
    
    # Predict
    prediction = gesture_model.predict(features_array)[0]
    
    return prediction

def reset_history():
    """Call this when starting a new session."""
    landmark_history.clear()

# Test
if __name__ == "__main__":
    print("Testing gesture detection with motion tracking...")
    print(f"Model loaded: {gesture_model is not None}")
    print(f"History buffer size: {landmark_history.maxlen} frames")