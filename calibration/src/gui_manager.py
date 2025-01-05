import cv2
import numpy as np
import sys
import os

# Ajout du chemin parent pour l'import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COLORS, WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT

class CalibrationGUI:
    """
    Interface graphique pour la calibration des couleurs.
    """
    def __init__(self, calibrator):
        self.calibrator = calibrator
        self.current_frame = None
        self.drawing = False
        self.roi_start = None
        self.roi_end = None
        
    def run(self, video_source=0):
        """
        Lance l'interface de calibration.
        """
        cap = cv2.VideoCapture(video_source)
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)
        
        paused = True  # La vidéo commence en pause
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                if not hasattr(self, 'current_display_frame'):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    self.current_display_frame = frame
                frame = self.current_display_frame.copy()
                
            frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
            self.current_frame = frame.copy()
            
            # Affichage des instructions
            self._draw_instructions(frame)
            
            # Affichage du ROI en cours de dessin
            if self.drawing:
                cv2.rectangle(frame, self.roi_start, 
                            (self.roi_end[0], self.roi_end[1]),
                            (0, 255, 0), 2)
            
            # Afficher l'état de lecture
            status = "PAUSE" if paused else "LECTURE"
            cv2.putText(frame, status, (WINDOW_WIDTH - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow(WINDOW_NAME, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # Barre d'espace pour pause/lecture
                paused = not paused
            elif key == ord('n') and paused:  # Image suivante
                ret, frame = cap.read()
                if ret:
                    self.current_display_frame = frame
            self._handle_key(key)
            
        cap.release()
        cv2.destroyAllWindows()
        
    def _mouse_callback(self, event, x, y, flags, param):
        """Gestion des événements souris."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.roi_start = (x, y)
            self.roi_end = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.roi_end = (x, y)
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            # Création du ROI final
            x1, y1 = self.roi_start
            x2, y2 = self.roi_end
            roi = (min(x1, x2), min(y1, y2), 
                  abs(x2-x1), abs(y2-y1))
            self._handle_roi(roi) 

    def _handle_roi(self, roi):
        """
        Gère la sélection d'une région d'intérêt.
        """
        # Afficher un menu pour choisir la couleur
        print("\nSélectionnez la couleur pour cet échantillon:")
        for color_id, color_info in COLORS.items():
            print(f"- Appuyez sur '{color_info['key']}' pour {color_info['name']}")
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            for color_id, color_info in COLORS.items():
                if chr(key) == color_info['key']:
                    self.calibrator.add_sample(color_id, self.current_frame, roi)
                    self.calibrator.calibrate_color(color_id)
                    print(f"Échantillon ajouté pour la couleur {color_info['name']}")
                    return
            if key == 27:  # ESC
                return

    def _draw_instructions(self, frame):
        """
        Affiche les instructions à l'écran.
        """
        y_pos = 30
        cv2.putText(frame, "Instructions:", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_pos += 25
        cv2.putText(frame, "1. Dessinez un rectangle avec la souris", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_pos += 25
        cv2.putText(frame, "2. Choisissez une couleur avec les touches:", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        for color_id, color_info in COLORS.items():
            y_pos += 25
            text = f"   {color_info['key']}: {color_info['name']}"
            cv2.putText(frame, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_info['display_color'], 1)
        
        y_pos += 25
        cv2.putText(frame, "Contrôles:", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_pos += 25
        cv2.putText(frame, "ESPACE: Pause/Lecture", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_pos += 25
        cv2.putText(frame, "N: Image suivante", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_pos += 25
        cv2.putText(frame, "ESC: Quitter", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def _handle_key(self, key):
        """
        Gère les touches pressées pendant la calibration.
        """
        if key == ord('c'):  # Effacer la sélection actuelle
            self.roi_start = None
            self.roi_end = None
            self.drawing = False 