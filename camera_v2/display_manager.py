import cv2
from config import frame_delay

class DisplayManager:
    def __init__(self):
        self.window_name = "Person Tracking"
        self.is_paused = False
        self.last_frame = None
        self.people_count = {"in": 0, "out": 0}
        self.tracked_paths = {}  # Pour stocker les trajectoires

    def draw_detection(self, frame, box, id, conf):
        # Dessiner le rectangle de détection
        color = (0, int(255 * conf), 0)
        x1, y1, x2, y2 = map(int, box)  # Assurez-vous que les coordonnées sont des entiers
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Stocker et dessiner la trajectoire
        center = ((x1 + x2) // 2, y2)
        if id not in self.tracked_paths:
            self.tracked_paths[id] = []
        self.tracked_paths[id].append(center)
        
        # Limiter la longueur de la trajectoire
        if len(self.tracked_paths[id]) > 30:  # Garder seulement les 30 dernières positions
            self.tracked_paths[id] = self.tracked_paths[id][-30:]
        
        # Dessiner la trajectoire
        if len(self.tracked_paths[id]) > 1:
            for i in range(1, len(self.tracked_paths[id])):
                cv2.line(frame, 
                        self.tracked_paths[id][i-1],
                        self.tracked_paths[id][i], 
                        color, 2)

        # Afficher les informations
        info_text = f"ID:{id} {conf:.2f}"
        cv2.putText(frame, 
                   info_text, 
                   (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, 
                   color, 
                   2)

    def draw_crossing_line(self, frame, start_point, end_point):
        cv2.line(frame, start_point, end_point, (0, 0, 255), 2)

    def draw_stats(self, frame):
        # Afficher les statistiques
        stats_text = [
            f"Personnes entrées: {self.people_count['in']}",
            f"Personnes sorties: {self.people_count['out']}",
            f"Total: {self.people_count['in'] + self.people_count['out']}"
        ]
        
        y = 30
        for text in stats_text:
            cv2.putText(frame, text, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 25

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