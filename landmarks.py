# landmarks.py
import cv2
import mediapipe as mp

# Initialize MediaPipe (only once)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def extract_landmarks(frame): 
    """
    Takes a BGR frame from webcam.
    Returns dict with landmark positions, or None if no face detected.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    if not results.multi_face_landmarks:
        return None
    
    face = results.multi_face_landmarks[0]
    
    landmarks = {
        'nose_tip': (face.landmark[4].x, face.landmark[4].y, face.landmark[4].z),
        'chin': (face.landmark[152].x, face.landmark[152].y, face.landmark[152].z),
        'left_eye': (face.landmark[33].x, face.landmark[33].y, face.landmark[33].z),
        'right_eye': (face.landmark[263].x, face.landmark[263].y, face.landmark[263].z),
        'forehead': (face.landmark[10].x, face.landmark[10].y, face.landmark[10].z),
    }
    
    return landmarks

# Test it
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        landmarks = extract_landmarks(frame)
        if landmarks:
            print(f"Nose: {landmarks['nose_tip']}")
        else:
            print("No face")
        
        cv2.imshow("Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()