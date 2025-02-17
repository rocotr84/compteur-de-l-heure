import cv2
import os
import numpy as np
from ultralytics import YOLO
from video_processor import VideoProcessor
from display_manager import DisplayManager
from tracker import PersonTracker
from config import *
from detection_history import DetectionHistory
from config import DETECTION_MODE
import signal
import sys
import torch
import time

"""
Module principal de l'application de comptage de personnes.
Gère l'initialisation des composants, le traitement vidéo et la coordination
entre les différents modules (détection, tracking, affichage).
"""

# Variable globale pour le color_tracker
detection_tracker = None

def signal_handler(sig, frame):
    """Gestionnaire pour l'arrêt propre du programme"""
    print("\nSauvegarde des données et arrêt du programme...")
    if detection_tracker:
        # Enregistre les données restantes pour chaque personne suivie
        for person_id in list(detection_tracker.color_history.keys()):
            detection_tracker.record_crossing(person_id)
        del detection_tracker
    sys.exit(0)

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU détectée: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("GPU non disponible, utilisation du CPU")
    return device

def main():
    """
    Fonction principale du programme
    """
    global detection_tracker
    
    # Initialisation du gestionnaire de signaux
    signal.signal(signal.SIGINT, signal_handler)  # Pour Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Pour kill
    
    detection_tracker = DetectionHistory()
    print(f"Démarrage du suivi en mode {DETECTION_MODE}...")
    
    # Initialisation des composants


    tracker = PersonTracker()
    video_proc = VideoProcessor(output_width, output_height, desired_fps)
    display = DisplayManager()
    
    # Configuration du processeur vidéo
    video_proc.load_mask(mask_path)
    cap = video_proc.setup_video_capture(video_path)
    
    device = setup_device()
    # Passage du modèle sur GPU
    tracker.model = tracker.model.to(device)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = video_proc.process_frame(frame)
            tracked_persons = tracker.update(frame)
            
            # Liste des IDs à traiter après la boucle
            ids_to_process = []
            
            # Utilisation d'une copie du dictionnaire pour l'itération
            for person_id, person in list(tracker.persons.items()):
                # Mise à jour de la couleur
                if hasattr(person, 'value') and person.value is not None:
                    detection_tracker.update_color(person_id, person.value)
                
                # Vérification du franchissement de ligne
                if person.check_line_crossing(line_start, line_end):
                    ids_to_process.append(person_id)
            
            # Traitement des IDs après l'itération
            for person_id in ids_to_process:
                if person_id in tracker.persons:
                    print(f"!!! Ligne traversée par ID={person_id} !!!")
                    elapsed_time = display.draw_timer(frame)
                    
                    # Assurez-vous que la couleur ou le numéro est mis à jour ici
                    detected_value = detection_tracker.get_dominant_value(person_id)  # Récupérer la valeur dominante
                    detection_tracker.update_color(person_id, detected_value)  # Mettez à jour l'historique
                    
                    detection_tracker.record_crossing(person_id, elapsed_time)  # Enregistrer le passage
                    dominant_value = detection_tracker.get_dominant_value(person_id)
                    if dominant_value:
                        tracker.counter[dominant_value] += 1
                    tracker.mark_as_crossed(person_id)
                    print(f"Personne ID={person_id} marquée comme ayant traversé")
            
            # Affichage
            for person in tracked_persons:
                display.draw_person(frame, person)
            
            display.draw_crossing_line(frame, line_start, line_end)
            display.draw_counters(frame, tracker.counter)
            
            should_quit, elapsed_time = display.show_frame(frame)  # Récupération du temps écoulé
            if should_quit:
                break
                
    except Exception as e:
        print(f"Erreur dans la boucle principale : {e}")
        
    finally:
        print("Fermeture du programme...")
        if detection_tracker:
            # Sauvegarde finale des données avec le temps écoulé final
            final_elapsed_time = time.time() - display.start_time
            for person_id in list(detection_tracker.color_history.keys()):
                detection_tracker.record_crossing(person_id, final_elapsed_time)
        cap.release()
        display.release()

# Point d'entrée du programme
if __name__ == "__main__":
    main() 