# training/train.py
import json
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_data(data_type="gesture"):
    """Load all JSON files and prepare features with temporal sequences."""
    data_dir = os.path.join(os.path.dirname(__file__), f"data/{data_type}")
    
    X = []  # Features
    y = []  # Labels
    
    print(f"\n📂 Loading {data_type} data from: {data_dir}")
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    print(f"Found {len(files)} files")
    
    for filename in files:
        # Extract label from filename (e.g., "wave_12345.json" → "wave")
        label = filename.split('_')[0]
        
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r') as f:
            frames = json.load(f)
        
        print(f"  {label}: {len(frames)} frames")
        
        # For gestures: Create temporal sequences (sliding window)
        # For emotions: Just use individual frames
        if data_type == "gesture":
            # Use sliding window of 5 frames
            window_size = 5
            for i in range(len(frames) - window_size + 1):
                window = frames[i:i+window_size]
                landmarks_sequence = [frame['landmarks'] for frame in window]
                
                features = extract_features(landmarks_sequence, data_type)
                if features is not None:
                    X.append(features)
                    y.append(label)
        else:
            # Emotions: use single frames
            for frame in frames:
                landmarks_sequence = [frame['landmarks']]  # Wrap in list for consistency
                features = extract_features(landmarks_sequence, data_type)
                if features is not None:
                    X.append(features)
                    y.append(label)
    
    return np.array(X), np.array(y)

def extract_features(landmarks_sequence, data_type):
    """
    Convert landmarks to feature vector.
    For gestures: Uses temporal features (motion over time)
    For emotions: Uses spatial features only
    
    landmarks_sequence: List of landmark dicts from consecutive frames
    """
    
    if data_type == "gesture":
        # Use the LAST frame for spatial features
        landmarks = landmarks_sequence[-1]
        features = []
        
        # === SPATIAL FEATURES (current pose) ===
        
        # Shoulder positions (for shrug, arms crossed)
        if 'pose' in landmarks:
            pose = landmarks['pose']
            features.extend([pose['left_shoulder'][0], pose['left_shoulder'][1]])
            features.extend([pose['right_shoulder'][0], pose['right_shoulder'][1]])
            # Elbow positions (for arms crossed)
            features.extend([pose['left_elbow'][0], pose['left_elbow'][1]])
            features.extend([pose['right_elbow'][0], pose['right_elbow'][1]])
        else:
            features.extend([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        
        # Hand positions (for wave, thumbs, pointing)
        if 'hands' in landmarks and len(landmarks['hands']) > 0:
            hand = landmarks['hands'][0]
            features.extend([hand['wrist'][0], hand['wrist'][1]])
            features.extend([hand['thumb_tip'][0], hand['thumb_tip'][1]])
            features.extend([hand['index_tip'][0], hand['index_tip'][1]])
            features.extend([hand['middle_tip'][0], hand['middle_tip'][1]])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Head position (for nod/shake baseline)
        if 'face' in landmarks and 'nose_tip' in landmarks['face']:
            nose = landmarks['face']['nose_tip']
            chin = landmarks['face']['chin']
            features.extend([nose[0], nose[1], chin[0], chin[1]])
        else:
            features.extend([0.5, 0.5, 0.5, 0.5])
        
        # === TEMPORAL FEATURES (motion over time) ===
        
        if len(landmarks_sequence) >= 5:
            # Compare current frame to frame from 5 steps ago
            old_landmarks = landmarks_sequence[-5]
            new_landmarks = landmarks_sequence[-1]
            
            # Head motion (for nod/shake)
            if ('face' in old_landmarks and 'nose_tip' in old_landmarks['face'] and
                'face' in new_landmarks and 'nose_tip' in new_landmarks['face']):
                
                old_nose = old_landmarks['face']['nose_tip']
                new_nose = new_landmarks['face']['nose_tip']
                
                # Motion in X (left/right - head shake)
                motion_x = new_nose[0] - old_nose[0]
                # Motion in Y (up/down - head nod)
                motion_y = new_nose[1] - old_nose[1]
                
                features.extend([motion_x, motion_y])
            else:
                features.extend([0.0, 0.0])
            
            # Shoulder motion (for shrug detection)
            if ('pose' in old_landmarks and 'pose' in new_landmarks):
                old_left_sh_y = old_landmarks['pose']['left_shoulder'][1]
                new_left_sh_y = new_landmarks['pose']['left_shoulder'][1]
                shoulder_motion = new_left_sh_y - old_left_sh_y  # Negative = moving up
                features.append(shoulder_motion)
            else:
                features.append(0.0)
            
            # Hand motion (for wave detection)
            if ('hands' in old_landmarks and len(old_landmarks['hands']) > 0 and
                'hands' in new_landmarks and len(new_landmarks['hands']) > 0):
                
                old_wrist_x = old_landmarks['hands'][0]['wrist'][0]
                new_wrist_x = new_landmarks['hands'][0]['wrist'][0]
                hand_motion_x = new_wrist_x - old_wrist_x
                features.append(hand_motion_x)
            else:
                features.append(0.0)
        else:
            # Not enough frames for temporal features
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        return features
    
    elif data_type == "emotion":
        # Emotions use only the LAST frame (no motion needed)
        landmarks = landmarks_sequence[-1]
        
        if 'face' not in landmarks:
            return None
        
        features = []
        face = landmarks['face']
        
        # Flatten all face coordinates (x, y only, ignore z)
        for key in sorted(face.keys()):
            features.extend([face[key][0], face[key][1]])
        
        return features
    
    return None

def train_model(data_type="gesture"):
    """Train and save the classifier."""
    print("=" * 50)
    print(f"🧠 TRAINING {data_type.upper()} MODEL")
    print("=" * 50)
    
    # Load data
    X, y = load_data(data_type)
    print(f"\n✅ Loaded {len(X)} total samples")
    print(f"📊 Classes: {sorted(set(y))}")
    
    # Count samples per class
    print("\n📈 Samples per class:")
    for label in sorted(set(y)):
        count = sum(1 for l in y if l == label)
        print(f"  {label}: {count}")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n🔀 Split data:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")
    
    # Train
    print(f"\n⚙️  Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\n📊 RESULTS:")
    print(f"  Training Accuracy: {train_score:.1%}")
    print(f"  Testing Accuracy:  {test_score:.1%}")
    
    # Detailed report
    predictions = model.predict(X_test)
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_test, predictions))
    
    # Save model
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{data_type}_model.pkl")
    joblib.dump(model, model_path)
    
    print(f"\n✅ Model saved to: {model_path}")
    print("=" * 50)
    
    return test_score

if __name__ == "__main__":
    print("\n🤖 MODEL TRAINING SYSTEM")
    print("=" * 50)
    
    choice = input("\nTrain [G]esture or [E]motion model? (or [B]oth): ").strip().upper()
    
    if choice == 'G':
        train_model("gesture")
    elif choice == 'E':
        train_model("emotion")
    elif choice == 'B':
        gesture_acc = train_model("gesture")
        emotion_acc = train_model("emotion")
        print("\n" + "=" * 50)
        print("🎉 TRAINING COMPLETE!")
        print("=" * 50)
        print(f"Gesture Model Accuracy: {gesture_acc:.1%}")
        print(f"Emotion Model Accuracy: {emotion_acc:.1%}")
    else:
        print("❌ Invalid choice. Run again and type G, E, or B.")