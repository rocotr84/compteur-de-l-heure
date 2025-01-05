import cv2
from config import frame_delay

class DisplayManager:
    def __init__(self):
        self.window_name = "Person Tracking"
        self.is_paused = False
        self.last_frame = None

    def draw_detection(self, frame, box, id, conf):
        color = (0, int(255 * conf), 0)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(frame, f"Person {id} ({conf:.2f})", 
                    (box[0], box[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, 2)

    def draw_crossing_line(self, frame, start_point, end_point):
        cv2.line(frame, start_point, end_point, (0, 0, 255), 2)

    def show_frame(self, frame):
        if self.is_paused:
            cv2.putText(frame, "PAUSE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if self.last_frame is not None:
                frame = self.last_frame.copy()
        else:
            self.last_frame = frame.copy()

        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(frame_delay) & 0xFF
        
        if key == ord('p'):
            self.is_paused = not self.is_paused
        
        return key == ord('q') 