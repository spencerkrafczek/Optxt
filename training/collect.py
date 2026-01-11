# training/collect.py
import cv2
import sys
import os
import time
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from landmarks import extract_landmarks

def collect_data(label, duration=5, data_type="gesture"):
    """Records landmarks for gestures or emotions."""
    
    save_dir = os.path.join(os.path.dirname(__file__), f"data/{data_type}")
    os.makedirs(save_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    frames_data = []
    
    print(f"\n=== Recording: {label} ===")
    print("Get ready...")
    time.sleep(2)
    print("GO!")
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        
        landmarks = extract_landmarks(frame)
        if landmarks:
            frames_data.append({
                'timestamp': time.time() - start_time,
                'landmarks': landmarks
            })
            
            # Visual feedback
            h, w, _ = frame.shape
            if 'face' in landmarks and 'nose_tip' in landmarks['face']:
                nose = landmarks['face']['nose_tip']
                cv2.circle(frame, (int(nose[0] * w), int(nose[1] * h)), 8, (0, 255, 0), -1)
            
            if 'hands' in landmarks:
                for hand in landmarks['hands']:
                    wrist = hand['wrist']
                    cv2.circle(frame, (int(wrist[0] * w), int(wrist[1] * h)), 8, (0, 0, 255), -1)
        
        # Countdown
        remaining = duration - (time.time() - start_time)
        cv2.putText(frame, f"{label}: {remaining:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Recording', frame)
        cv2.waitKey(1)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save
    filename = f"{save_dir}/{label}_{int(time.time())}.json"
    with open(filename, 'w') as f:
        json.dump(frames_data, f)
    
    print(f"✓ Saved {len(frames_data)} frames to {filename}")

if __name__ == "__main__":
    print("=" * 50)
    print("DATA COLLECTION SYSTEM")
    print("=" * 50)
    
    choice = input("\nCollect [G]estures or [E]motions? ").strip().upper()
    
    if choice == 'G':
        gestures = ["headnod", "headshake", "shrug", "wave", "middlefinger", "neutral"]
        print("\n=== GESTURE COLLECTION ===")
        print("Instructions:")
        print("  headnod: Nod up and down")
        print("  headshake: Shake left and right")
        print("  shrug: Shrug shoulders")
        print("  wave: Wave your hand")
        print("  middlefinger: You know")
        print("  neutral: Sit still, hands down")
        
        for gesture in gestures:
            input(f"\nPress Enter to record '{gesture}'...")
            collect_data(gesture, duration=5, data_type="gesture")
    
    elif choice == 'E':
        emotions = ["happy", "sad", "angry", "shocked", "neutral"]
        print("\n=== EMOTION COLLECTION ===")
        print("Instructions:")
        print("  happy: Big smile!")
        print("  sad: Frown, look down")
        print("  angry: Furrowed brow, frown")
        print("  shocked: Eyes wide, mouth open")
        print("  neutral: Relaxed face")
        
        for emotion in emotions:
            input(f"\nPress Enter to record '{emotion}'...")
            collect_data(emotion, duration=5, data_type="emotion")
    
    else:
        print("Invalid choice. Run again and type G or E.")
    
    print("\n=== Done! ===")