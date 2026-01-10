from deepface import DeepFace
import cv2

def get_emotion(frame):
    """
    Analyzes the frame locally using DeepFace.
    Returns the dominant emotion (e.g., 'happy', 'sad', 'neutral').
    """
    try:
       
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        analysis = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)

        if isinstance(analysis, list):
            result = analysis[0]
        else:
            result = analysis

        emotion = result['dominant_emotion']
    
        return emotion

    except ValueError:
        return None
    except Exception as e:
        print(f"DeepFace Error: {e}")
        return None

# ... (Your existing code stays above this) ...

# --- ADD THIS TO THE BOTTOM ---
if __name__ == "__main__":
    print("--- Testing DeepFace (Local AI) ---")
    
    # 1. Open the camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    
    # 2. Grab a frame
    ret, frame = cap.read()
    
    if ret:
        print("Capturing frame...")
        # 3. Call your function
        emotion = get_emotion(frame)
        print(f"\nRESULT: The AI thinks you are: {emotion}")
        
        # Show the image so you know what it saw
        cv2.imshow("What the AI saw", frame)
        cv2.waitKey(0) # Press any key to close window
        cv2.destroyAllWindows()
    else:
        print("Failed to capture image.")
        
    cap.release()