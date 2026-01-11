# main.py
import cv2 
import time

from landmarks import extract_landmarks
from gestures import detect_gesture
from emotions import detect_emotion
from speech import say_interaction
from state_tracker import StateTracker

def main():
    print("=" * 50)
    print("OPTXT - INTERVIEW PRACTICE TOOL")
    print("=" * 50)

    # Setup webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not found")
        return
    
    print("✅ Webcam ready")

    # State tracker (prevents repeating same info)
    tracker = StateTracker()

    print("🎬 Running... Press 'q' to quit")
    print("=" * 50)

    last_announcement_time = time.time()
    announcement_cooldown = 4.0  # Only announce changes every 2 seconds
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break

        # Extract landmarks from current frame
        landmarks = extract_landmarks(frame)
        
        current_gesture = "no_data"
        current_emotion = "no_data"
        
        if landmarks:
            # Detect gesture and emotion
            current_gesture = detect_gesture(landmarks)
            current_emotion = detect_emotion(landmarks)
            
            # Check if anything changed
            current_time = time.time()
            if (current_time - last_announcement_time) >= announcement_cooldown:
                changed, message = tracker.update(current_gesture, current_emotion)
                
                # Inside the 'if changed:' block of main.py
                if changed:
                    print(f">>> {message}")
                    # Pass 'current_emotion' so speech.py knows which settings to use
                    say_interaction(message, emotion=current_emotion)
                    last_announcement_time = current_time
                if changed:
                    print(f">>> {message}")
                    say_interaction(message)
                    last_announcement_time = current_time
        
        # Visual feedback
        display_frame = frame.copy()
        
        # Draw current status
        cv2.putText(
            display_frame,
            f"Gesture: {current_gesture}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        cv2.putText(
            display_frame,
            f"Emotion: {current_emotion}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Show FPS
        cv2.putText(
            display_frame,
            "Press 'q' to quit",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.imshow("Interview Practice", display_frame)
                
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Thanks for practicing!")

if __name__ == "__main__":
    main()