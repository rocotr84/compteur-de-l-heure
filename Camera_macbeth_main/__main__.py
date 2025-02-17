import cv2
import os
import numpy as np
from ultralytics import YOLO
from config import *
import signal
import sys
import torch
import time
from video_processor import (
    init_video_processor,
    load_mask,
    setup_video_capture,
    process_frame
) 
from display_manager import (
    init_display,
    draw_person,
    draw_counters,
    draw_crossing_line,
    draw_timer,
    show_frame,
    release
)
from tracker import (
    create_tracker,
    update_tracker,
    check_line_crossing,
    mark_as_crossed
)

from detection_history import (
    cleanup,
    init_detection_history,
    update_color,
    get_dominant_value,
    record_crossing
)



"""
Module principal de l'application de comptage de personnes.
Gère l'initialisation des composants, le traitement vidéo et la coordination
entre les différents modules (détection, tracking, affichage).
"""

# Variable globale pour le color_tracker


def signal_handler(sig, frame):
    """Gestionnaire pour l'arrêt propre du programme"""
    print("\nSauvegarde des données et arrêt du programme...")
    cleanup()  # Nouveau nom de la fonction
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
    
    init_detection_history(output_file)  
    print(f"Démarrage du suivi en mode {DETECTION_MODE}...")
    
    # Initialisation du processeur vidéo
    init_video_processor(output_width, output_height, desired_fps)
    load_mask(mask_path)

    cap = setup_video_capture(video_path)
    
    # Création du tracker
    tracker_state = create_tracker()
    init_display()  # Initialisation de l'affichage
    
    device = setup_device()
    tracker_state['model'] = tracker_state['model'].to(device)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = process_frame(frame, cache_file, detect_squares)
            tracked_persons = update_tracker(tracker_state, frame)
            
            ids_to_process = []
            
            for person in tracked_persons:
                person_id = person['id']
                if person['value'] is not None:
                    update_color(person_id, person['value'])  # Appel direct de la fonction
                
                if check_line_crossing(person, line_start, line_end):
                    ids_to_process.append(person_id)
            
            for person_id in ids_to_process:
                if person_id in tracker_state['persons']:
                    print(f"!!! Ligne traversée par ID={person_id} !!!")
                    elapsed_time = draw_timer(frame)
                    
                    detected_value = get_dominant_value(person_id) 
                    update_color(person_id, detected_value)  
                    
                    record_crossing(person_id, elapsed_time) 
                    dominant_value = get_dominant_value(person_id) 
                    if dominant_value:
                        tracker_state['counter'][dominant_value] += 1
                    mark_as_crossed(tracker_state, person_id)
                    print(f"Personne ID={person_id} marquée comme ayant traversé")
            
            for person in tracked_persons:
                draw_person(frame, person)
            
            draw_crossing_line(frame, line_start, line_end)
            draw_counters(frame, tracker_state['counter'])
            
            should_quit, elapsed_time = show_frame(frame)
            if should_quit:
                break
                
    except Exception as e:
        print(f"Erreur dans la boucle principale : {e}")
        
    finally:
        print("Fermeture du programme...")
        cleanup()  # Appel de la nouvelle fonction de nettoyage
        cap.release()
        release()  # Appel de la fonction release au lieu de display.release()

# Point d'entrée du programme
if __name__ == "__main__":
    main() 