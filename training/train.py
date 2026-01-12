# training/train.py
import json
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_data(data_type="gesture"):
    """Load all JSON files and prepare features (single frame, no motion)."""
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
        
        # Extract features from EACH FRAME (no temporal sequences)
        for frame in frames:
            landmarks = frame['landmarks']
            features = extract_features(landmarks, data_type)
            if features is not None:
                X.append(features)
                y.append(label)
    
    return np.array(X), np.array(y)

def extract_features(landmarks, data_type):
    """Convert landmarks to feature vector (STATIC - NO MOTION)."""
    features = []
    
    if data_type == "gesture":
        # Gesture features: pose (shoulders) + hands
        
        # Shoulder positions (for shrug detection)
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
    
    elif data_type == "emotion":
        # Emotion features: face landmarks only
        if 'face' not in landmarks:
            return None
        
        face = landmarks['face']
        # Flatten all face coordinates (x, y only, ignore z)
        for key in sorted(face.keys()):  # Sort for consistency
            features.extend([face[key][0], face[key][1]])
        
        return features
    
    return None

def train_model(data_type="gesture"):
    """Train and save the classifier."""
    print("=" * 50)
    print(f"🧠 TRAINING {data_type.upper()} MODEL (STATIC POSES)")
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