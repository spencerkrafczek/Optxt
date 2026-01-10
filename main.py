import cv2 
import time 
from collections import deque

#importing everyone elses codes
from landmarks import extract_landmarks
from gestures import detect_gesture
from emotions import detect_emotion
from speech import speak
from state_tracker import StateTracker

def main():
    print("=" * 50)
    print("OPTXT - Social Cue Detection for the Blind")
    print("=" * 50)


    # Sets up the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera couldn't be found")
        return
    
    print("Webcam found")

    #Sets up the buffer - delay of last 60 frames of data to detect gestures
    data_buffer = deque(maxlen=60)

    #State tracker this prevents the same info being repeated multiple times
    tracker = StateTracker(cooldown=1.5)

    #Timing for intervals
    last_analysis_time = 0
    analysis_interval = 0.5

    print("Code is running. Press \'q\' to quit.")
    print("=" * 50)

    last_analysis_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to find face")
            break

        current_time = time.time()
        # --- PERSON A'S WORK GOES HERE (Landmarks) ---
        landmarks = extract_landmarks(frame)
        if landmarks:
            landmark_buffer.append(landmarks)

        if (current_time - last_analysis_time) >= analysis_interval:
            last_analysis_time = current_time

        gesture = detect_gesture(list(landmark_buffer))

        # --- PERSON B'S WORK GOES HERE (Analysis) ---
        emotion = detect_emotion(frame)

        changes = tracker.get_changes(gesture, emotion)

        if changes:
            output = ", ".join(changes)
            print(f">>> {output}")
            speak(output)


        # Analyze every 1 second so we don't lag
        if (time.time() - last_analysis_time) > 1.0:
            last_analysis_time = time.time()
            
        #The visual feedback
        display_frame = frame.copy()

        
        cv2.putText(
            display_frame,
            f"Gesture: {tracker.previous_gesture or 'detecting...'}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        cv2.putText(
            display_frame,
            f"Emotion: {tracker.previous_emotion or 'detecting...'}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        cv2.putText(
            display_frame,
            f"Buffer: {len(landmark_buffer)}/60",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        cv2.imshow("Optxt", display_frame)
                
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()