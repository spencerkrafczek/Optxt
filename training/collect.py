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
            
            # === FULL VISUAL FEEDBACK ===
            h, w, _ = frame.shape
            
            # Draw FACE (green dots)
            if 'face' in landmarks:
                face = landmarks['face']
                for point_name, (x, y, z) in face.items():
                    cv2.circle(frame, (int(x * w), int(y * h)), 3, (0, 255, 0), -1)
            
            # Draw POSE (blue skeleton)
            if 'pose' in landmarks:
                pose = landmarks['pose']
                
                # Draw shoulder line
                left_sh = (int(pose['left_shoulder'][0] * w), int(pose['left_shoulder'][1] * h))
                right_sh = (int(pose['right_shoulder'][0] * w), int(pose['right_shoulder'][1] * h))
                cv2.line(frame, left_sh, right_sh, (255, 0, 0), 3)
                
                # Draw left arm
                left_elb = (int(pose['left_elbow'][0] * w), int(pose['left_elbow'][1] * h))
                left_wrist = (int(pose['left_wrist'][0] * w), int(pose['left_wrist'][1] * h))
                cv2.line(frame, left_sh, left_elb, (255, 0, 0), 2)
                cv2.line(frame, left_elb, left_wrist, (255, 0, 0), 2)
                
                # Draw right arm
                right_elb = (int(pose['right_elbow'][0] * w), int(pose['right_elbow'][1] * h))
                right_wrist = (int(pose['right_wrist'][0] * w), int(pose['right_wrist'][1] * h))
                cv2.line(frame, right_sh, right_elb, (255, 0, 0), 2)
                cv2.line(frame, right_elb, right_wrist, (255, 0, 0), 2)
                
                # Draw joint dots
                for point in [left_sh, right_sh, left_elb, right_elb, left_wrist, right_wrist]:
                    cv2.circle(frame, point, 6, (255, 0, 0), -1)
            
            # Draw HANDS (red dots)
            if 'hands' in landmarks:
                for hand in landmarks['hands']:
                    for point_name, (x, y, z) in hand.items():
                        cv2.circle(frame, (int(x * w), int(y * h)), 6, (0, 0, 255), -1)
        
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
    print("SOCIAL CUE DATA COLLECTION")
    print("=" * 50)
    
    choice = input("\nCollect [G]estures or [E]motions? ").strip().upper()
    
    if choice == 'G':
        gestures = ["wave", "thumbsup", "thumbsdown", "pointingatyou", "middlefinger", "shrug", "headnod", "headshake", "neutral"]
        print("\n=== GESTURE COLLECTION ===")
        print("Instructions:")
        print("  wave: Wave hand side to side CONTINUOUSLY for 5 seconds")
        print("  thumbsup: Hold thumbs up gesture steady")
        print("  thumbsdown: Hold thumbs down gesture steady")
        print("  pointingatyou: Point index finger at camera, hold steady")
        print("  middlefinger: Hold middle finger up steady")
        print("  shrug: Shrug shoulders UP and hold for 5 seconds")
        print("  headnod: Nod head up and down CONTINUOUSLY for 5 seconds")
        print("  headshake: Shake head left and right CONTINUOUSLY for 5 seconds")
        print("  neutral: Sit still, hands down, no movement at all")
        
        for gesture in gestures:
            input(f"\nPress Enter to record '{gesture}'...")
            collect_data(gesture, duration=5, data_type="gesture")
    
    elif choice == 'E':
        emotions = ["happy", "angry", "shocked", "neutral"]
        print("\n=== EMOTION COLLECTION ===")
        print("Instructions:")
        print("  happy: BIG smile, show teeth, raise cheeks")
        print("  angry: REALLY furrow eyebrows, frown hard, tighten jaw")
        print("  shocked: Eyes SUPER wide, eyebrows HIGH, mouth in big O shape")
        print("  neutral: Completely relaxed, resting face, no expression")
        
        for emotion in emotions:
            input(f"\nPress Enter to record '{emotion}'...")
            collect_data(emotion, duration=5, data_type="emotion")
    
    else:
        print("Invalid choice. Run again and type G or E.")
    
    print("\n=== Done! ===")
