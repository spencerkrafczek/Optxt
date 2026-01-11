import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

# Try to import from main project, fall back to local copy
try:
    from landmarks import extract_hand_landmarks
except ImportError:
    from training.landmarks import extract_hand_landmarks

class DataCollector:
    def __init__(self, data_type='gesture'):
        """
        data_type: 'gesture' or 'emotion'
        """
        self.data_type = data_type
        self.data_dir = f'training/data/{data_type}'
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        self.current_label = None
        self.samples = []
        
    def collect_gesture_sample(self, frame, label):
        """Collect gesture sample (hand landmarks)"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = extract_hand_landmarks(results.multi_hand_landmarks[0])
            self.samples.append({
                'landmarks': landmarks,
                'label': label,
                'timestamp': datetime.now().isoformat()
            })
            return True
        return False
    
    def collect_emotion_sample(self, frame, label):
        """Collect emotion sample (face landmarks)"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            landmarks = []
            for lm in face.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            self.samples.append({
                'landmarks': landmarks,
                'label': label,
                'timestamp': datetime.now().isoformat()
            })
            return True
        return False
    
    def save_samples(self):
        """Save collected samples to file"""
        if not self.samples:
            print("No samples to save!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.data_dir}/{self.current_label}_{timestamp}.npy"
        
        # Convert to numpy array
        data = {
            'landmarks': [s['landmarks'] for s in self.samples],
            'labels': [s['label'] for s in self.samples],
            'timestamps': [s['timestamp'] for s in self.samples]
        }
        
        np.save(filename, data)
        print(f"Saved {len(self.samples)} samples to {filename}")
        self.samples = []

def main():
    print("=" * 50)
    print("DATA COLLECTION TOOL")
    print("=" * 50)
    print("\nChoose data type:")
    print("1. Gesture (hand landmarks)")
    print("2. Emotion (face landmarks)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    data_type = 'gesture' if choice == '1' else 'emotion'
    
    collector = DataCollector(data_type=data_type)
    
    if data_type == 'gesture':
        print("\nGesture labels: rock, paper, scissors, none")
        label = input("Enter gesture label: ").strip().lower()
    else:
        print("\nEmotion labels: happy, sad, angry, neutral, surprised")
        label = input("Enter emotion label: ").strip().lower()
    
    collector.current_label = label
    
    print(f"\nCollecting {data_type} data for: {label}")
    print("Press 'q' to stop and save")
    print("Press SPACE to capture sample")
    print("\nStarting camera...")
    
    cap = cv2.VideoCapture(0)
    sample_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        
        # Add UI
        cv2.putText(display, f"Label: {label}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, f"Samples: {sample_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "SPACE = Capture | Q = Save & Quit", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Data Collection', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            if data_type == 'gesture':
                success = collector.collect_gesture_sample(frame, label)
            else:
                success = collector.collect_emotion_sample(frame, label)
            
            if success:
                sample_count += 1
                print(f"✓ Captured sample {sample_count}")
            else:
                print(f"✗ Failed - no {data_type} detected")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    collector.save_samples()
    print(f"\nTotal samples collected: {sample_count}")

if __name__ == "__main__":
    main()