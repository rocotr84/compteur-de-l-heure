import cv2
import os
import numpy as np
from ultralytics import YOLO
from video_processor import VideoProcessor
from display_manager import DisplayManager
from tracker import PersonTracker
from config import *

"""
Module principal de l'application de comptage de personnes.
Gère l'initialisation des composants, le traitement vidéo et la coordination
entre les différents modules (détection, tracking, affichage).
"""

def main():
    """
    Fonction principale du programme
    """
    # Initialisation des composants
    tracker = PersonTracker()
    video_proc = VideoProcessor(output_width, output_height, desired_fps)
    display = DisplayManager()
    
    # Configuration du processeur vidéo
    video_proc.load_mask(mask_path)
    cap = video_proc.setup_video_capture(video_path)
    
    try:
        while True:
            # Lecture d'une nouvelle frame
            ret, frame = cap.read()
            if not ret:
                break
                
            # Prétraitement de la frame
            frame = video_proc.process_frame(frame)
            
            # Mise à jour du tracking
            tracked_persons = tracker.update(frame)
            
            # Traitement de chaque personne détectée
            for person in tracked_persons:
                display.draw_person(frame, person)
                # Vérification du franchissement de la ligne
                if person.check_line_crossing(line_start, line_end):
                    tracker.counter[person.color] += 1
                    
            # Affichage des éléments visuels
            display.draw_crossing_line(frame, line_start, line_end)
            display.draw_counters(frame, tracker.counter)
            
            # Affichage ou enregistrement de la frame
            if display.show_frame(frame):
                break
                
    finally:
        # Nettoyage des ressources
        cap.release()
        display.release()

# Point d'entrée du programme
if __name__ == "__main__":
    main() 