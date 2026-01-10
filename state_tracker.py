class StateTracker:
    def __init__(self):
        self.current_gesture = None
        self.current_emotion = None

    def update(self, new_gesture, new_emotion):
        change_detected = False
        message = ""

        if new_gesture and new_gesture != self.current_gesture:
            self.current_gesture = new_gesture
            change_detected = True
        
        if new_emotion and new_emotion != self.current_emotion:
            self.current_emotion = new_emotion
            change_detected = True

        if change_detected:
            message = f"Detected {self.current_gesture} and feeling {self.current_emotion}"
        
        return change_detected, message