# training/collect.py
import cv2
import sys
import os
import time
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from landmarks import extract_landmarks

def collect_gesture_data(gesture_name, duration=5):
    """Records landmarks for a gesture for 5 seconds."""
    
    save_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(save_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    frames_data = []
    
    print(f"\n=== Recording: {gesture_name} ===")
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
            
            # === GREEN DOT CODE ===
            h, w, _ = frame.shape                          # Get frame dimensions (height, width)
            nose = landmarks['nose_tip']                   # Get nose coordinates (0 to 1 range)
            x_px = int(nose[0] * w)                        # Convert to pixel position (0 to 1) * width
            y_px = int(nose[1] * h)                        # Convert to pixel position (0 to 1) * height
            cv2.circle(frame, (x_px, y_px), 8, (0, 255, 0), -1)  # Draw green circle, radius 8, filled (-1)
        
        # Show countdown on screen
        remaining = duration - (time.time() - start_time)
        cv2.putText(frame, f"{gesture_name}: {remaining:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Recording', frame)
        cv2.waitKey(1)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save to file
    filename = f"{save_dir}/{gesture_name}_{int(time.time())}.json"
    with open(filename, 'w') as f:
        json.dump(frames_data, f)
    
    print(f"Saved {len(frames_data)} frames to {filename}")

if __name__ == "__main__":
    gestures = ["nodding", "shaking", "fidgeting", "still"]
    
    print("=== Gesture Data Collection ===")
    
    for gesture in gestures:
        input(f"\nPress Enter to record '{gesture}'...")
        collect_gesture_data(gesture, duration=5)
    
    print("\n=== Done! ===")