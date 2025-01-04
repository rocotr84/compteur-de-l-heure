import cv2
import os
import numpy as np
from ultralytics import YOLO
from video_processor import VideoProcessor
from display_manager import DisplayManager
from tracker import PersonTracker
from config import *  # Importez toutes les constantes

"""
Module principal de l'application de comptage de personnes.
Gère l'initialisation des composants, le traitement vidéo et la coordination
entre les différents modules (détection, tracking, affichage).
"""

def process_yolo_detections(model, frame):
    """
    Traite les détections YOLO sur une frame
    Args:
        model (YOLO): Modèle YOLO chargé
        frame (np.array): Image à analyser
    Returns:
        tuple: (detections, confidences)
            - detections: liste des boîtes englobantes [x1, y1, x2, y2]
            - confidences: liste des scores de confiance associés
    """
    results = model(frame)[0]
    detections = []
    confidences = []
    
    # Filtrage des détections pour ne garder que les personnes avec une confiance suffisante
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = r
        if score > MIN_CONFIDENCE and int(class_id) == 0:  # 0 = personne
            detections.append([x1, y1, x2, y2])
            confidences.append(score)
            
    return detections, confidences

def main():
    """
    Fonction principale du programme
    Initialise les composants et exécute la boucle principale de traitement vidéo
    """
    # Initialisation des composants
    model = YOLO(modele_path)                # Modèle de détection
    tracker = PersonTracker()                # Gestionnaire de tracking
    video_proc = VideoProcessor(output_width, output_height, desired_fps)  # Processeur vidéo
    display = DisplayManager()               # Gestionnaire d'affichage
    
    # Configuration du processeur vidéo
    video_proc.load_mask(mask_path)
    cap = video_proc.setup_video_capture(video_path)
    
    # Boucle principale de traitement
    while True:
        # Lecture d'une nouvelle frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Prétraitement de la frame
        frame = video_proc.process_frame(frame)
        
        # Détection et tracking des personnes
        detections, confidences = process_yolo_detections(model, frame)
        tracked_persons = tracker.update(detections, frame, confidences)
        
        # Affichage des résultats
        for person in tracked_persons:
            display.draw_person(frame, person)
            # Vérification du franchissement de la ligne
            if person.check_line_crossing(line_start, line_end):
                tracker.counter[person.color] += 1
                
        # Affichage des éléments visuels
        display.draw_crossing_line(frame, line_start, line_end)
        display.draw_counters(frame, tracker.counter)
        
        # Gestion de la sortie
        if display.show_frame(frame):
            break
            
    # Nettoyage des ressources
    cap.release()
    cv2.destroyAllWindows()

# Point d'entrée du programme
if __name__ == "__main__":
    main() 