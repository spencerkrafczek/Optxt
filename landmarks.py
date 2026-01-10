# landmarks.py
# Run: python3.11 landmarks.py
# Quit: press 'q' in the window

import cv2
import mediapipe as mp

def main():
    print("Starting webcam + MediaPipe FaceMesh test...")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam with VideoCapture(0).")
        print("Try cv2.VideoCapture(1) if you have multiple cameras.")
        return

    print("Webcam opened! Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            # nose tip-ish landmark (index 4 is common in examples)
            nose_tip = face.landmark[4]

            h, w, _ = frame.shape
            x_px = int(nose_tip.x * w)
            y_px = int(nose_tip.y * h)

            cv2.circle(frame, (x_px, y_px), 6, (0, 255, 0), -1)
            print(f"Nose tip: x={nose_tip.x:.3f}, y={nose_tip.y:.3f}, z={nose_tip.z:.3f}")
        else:
            print("No face detected")

        cv2.imshow("FaceMesh Test (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("Test complete!")

if __name__ == "__main__":
    main()