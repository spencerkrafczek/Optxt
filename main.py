import cv2 
import time 
from collections import deque

def main():
    print("Running Optxt...")

    # Sets up the webcam
    cap = cv2.VideoCapture(0)

    #Sets up the buffer - delay of last 60 frames of data to detect gestures
    data_buffer = deque(maxlen=60)

    last_analysis_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- PERSON A'S WORK GOES HERE (Landmarks) ---
        # landmarks = PersonA.get_landmarks(frame)
        # data_buffer.append(landmarks)

        # --- PERSON B'S WORK GOES HERE (Analysis) ---
        # Analyze every 1 second so we don't lag

        if (time.time() - last_analysis_time) > 1.0:
            last_analysis_time = time.time()
            
        #The visual feedback
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Buffer: {len(data_buffer)}/60",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        cv2.imshow("Optxt", display_frame)
                
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()