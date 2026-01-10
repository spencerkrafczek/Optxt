# landmarks.py
import cv2
import mediapipe as mp

# Initialize ALL MediaPipe models (only once)
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

face_mesh_model = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

pose_model = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def extract_landmarks(frame): 
    """
    Takes a BGR frame from webcam.
    Returns dict with face, pose, and hand landmarks, or None if nothing detected.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process all three models
    face_results = face_mesh_model.process(rgb)
    pose_results = pose_model.process(rgb)
    hand_results = hands_model.process(rgb)
    
    landmarks = {}
    
    # === FACE LANDMARKS ===
    if face_results.multi_face_landmarks:
        face = face_results.multi_face_landmarks[0]
        landmarks['face'] = {
            'nose_tip': (face.landmark[4].x, face.landmark[4].y, face.landmark[4].z),
            'chin': (face.landmark[152].x, face.landmark[152].y, face.landmark[152].z),
            'left_eye': (face.landmark[33].x, face.landmark[33].y, face.landmark[33].z),
            'right_eye': (face.landmark[263].x, face.landmark[263].y, face.landmark[263].z),
            'forehead': (face.landmark[10].x, face.landmark[10].y, face.landmark[10].z),
        }
    
    # === POSE LANDMARKS (body/shoulders) ===
    if pose_results.pose_landmarks:
        pose_lm = pose_results.pose_landmarks.landmark
        landmarks['pose'] = {
            'left_shoulder': (pose_lm[11].x, pose_lm[11].y, pose_lm[11].z),
            'right_shoulder': (pose_lm[12].x, pose_lm[12].y, pose_lm[12].z),
            'left_elbow': (pose_lm[13].x, pose_lm[13].y, pose_lm[13].z),
            'right_elbow': (pose_lm[14].x, pose_lm[14].y, pose_lm[14].z),
            'left_wrist': (pose_lm[15].x, pose_lm[15].y, pose_lm[15].z),
            'right_wrist': (pose_lm[16].x, pose_lm[16].y, pose_lm[16].z),
        }
    
    # === HAND LANDMARKS ===
    if hand_results.multi_hand_landmarks:
        landmarks['hands'] = []
        for hand_landmarks in hand_results.multi_hand_landmarks:
            hand_data = {
                'wrist': (hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z),
                'thumb_tip': (hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y, hand_landmarks.landmark[4].z),
                'index_tip': (hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y, hand_landmarks.landmark[8].z),
                'middle_tip': (hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y, hand_landmarks.landmark[12].z),
                'ring_tip': (hand_landmarks.landmark[16].x, hand_landmarks.landmark[16].y, hand_landmarks.landmark[16].z),
                'pinky_tip': (hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y, hand_landmarks.landmark[20].z),
            }
            landmarks['hands'].append(hand_data)
    
    return landmarks if landmarks else None

# Test it
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        landmarks = extract_landmarks(frame)
        
        if landmarks:
            h, w, _ = frame.shape
            
            # === DRAW FACE (green dots) ===
            if 'face' in landmarks:
                face = landmarks['face']
                for point_name, (x, y, z) in face.items():
                    cv2.circle(frame, (int(x * w), int(y * h)), 5, (0, 255, 0), -1)
            
            # === DRAW POSE (stick figure - blue) ===
            if 'pose' in landmarks:
                pose = landmarks['pose']
                
                # Draw shoulder line
                left_sh = (int(pose['left_shoulder'][0] * w), int(pose['left_shoulder'][1] * h))
                right_sh = (int(pose['right_shoulder'][0] * w), int(pose['right_shoulder'][1] * h))
                cv2.line(frame, left_sh, right_sh, (255, 0, 0), 3)
                
                # Draw left arm
                left_elb = (int(pose['left_elbow'][0] * w), int(pose['left_elbow'][1] * h))
                left_wrist = (int(pose['left_wrist'][0] * w), int(pose['left_wrist'][1] * h))
                cv2.line(frame, left_sh, left_elb, (255, 0, 0), 3)
                cv2.line(frame, left_elb, left_wrist, (255, 0, 0), 3)
                
                # Draw right arm
                right_elb = (int(pose['right_elbow'][0] * w), int(pose['right_elbow'][1] * h))
                right_wrist = (int(pose['right_wrist'][0] * w), int(pose['right_wrist'][1] * h))
                cv2.line(frame, right_sh, right_elb, (255, 0, 0), 3)
                cv2.line(frame, right_elb, right_wrist, (255, 0, 0), 3)
                
                # Draw joint dots
                for point in [left_sh, right_sh, left_elb, right_elb, left_wrist, right_wrist]:
                    cv2.circle(frame, point, 8, (255, 0, 0), -1)
            
            # === DRAW HANDS (red dots) ===
            if 'hands' in landmarks:
                for hand in landmarks['hands']:
                    for point_name, (x, y, z) in hand.items():
                        cv2.circle(frame, (int(x * w), int(y * h)), 8, (0, 0, 255), -1)
        
        cv2.imshow("Test - Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()