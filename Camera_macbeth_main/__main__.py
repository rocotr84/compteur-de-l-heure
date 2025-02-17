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

from macbeth_color_and_rectangle_detector import get_average_colors



"""
Module principal de l'application de comptage de personnes.

Ce module coordonne les différentes composantes du système :
- Détection et suivi des personnes
- Traitement vidéo et correction des couleurs
- Gestion de l'affichage et des compteurs
- Enregistrement des données de passage

Le système utilise une charte Macbeth pour la calibration des couleurs
et permet le comptage de personnes en fonction de caractéristiques détectées.
"""

# Variable globale pour le color_tracker


def signal_handler(sig, frame):
    """
    Gestionnaire pour l'arrêt propre du programme.
    
    Assure la sauvegarde des données et la fermeture correcte des ressources
    lors de l'interruption du programme (Ctrl+C ou SIGTERM).
    
    Args:
        sig: Signal reçu
        frame: Frame d'exécution courante
    """
    print("\nSauvegarde des données et arrêt du programme...")
    cleanup()
    sys.exit(0)

def setup_device():
    """
    Configure le dispositif de calcul (GPU/CPU) pour le traitement.
    
    Détecte la présence d'un GPU compatible CUDA et configure
    l'environnement en conséquence.
    
    Returns:
        torch.device: Dispositif de calcul configuré (GPU ou CPU)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU détectée: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("GPU non disponible, utilisation du CPU")
    return device

def initialize():
    """
    Initialise tous les composants nécessaires au fonctionnement du programme.
    
    Cette fonction :
    1. Configure les gestionnaires de signaux
    2. Initialise l'historique de détection
    3. Configure le processeur vidéo et charge le masque
    4. Initialise le tracker et l'affichage
    5. Configure le dispositif de calcul
    6. Effectue la détection initiale des couleurs Macbeth
    
    Returns:
        tuple: (
            cv2.VideoCapture: Capture vidéo configurée,
            np.array: Masque chargé,
            dict: État du tracker initialisé
        )
    """
    # Initialisation du gestionnaire de signaux
    signal.signal(signal.SIGINT, signal_handler)  # Pour Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Pour kill

    init_detection_history(output_file)
    print(f"Démarrage du suivi en mode {DETECTION_MODE}...")

    # Initialisation du processeur vidéo
    init_video_processor(output_width, output_height, desired_fps)
    mask = load_mask(mask_path)

    cap = setup_video_capture(video_path)

    # Création du tracker
    tracker_state = create_tracker()
    init_display()  # Initialisation de l'affichage

    device = setup_device()
    tracker_state['model'] = tracker_state['model'].to(device)

    # Détection des carrés et des couleurs moyennes de la charte Macbeth
    _, frame = cap.read()
    if frame is not None:
        get_average_colors(frame, cache_file, detect_squares = True)
    else:
        print("Impossible de lire la première frame de la vidéo.")

    return cap, mask, tracker_state

def main():
    """
    Fonction principale du programme.
    
    Gère la boucle principale de traitement qui :
    1. Lit les frames de la vidéo
    2. Traite chaque frame (correction couleur, détection)
    3. Met à jour le suivi des personnes
    4. Détecte les traversées de ligne
    5. Met à jour les compteurs et l'affichage
    6. Enregistre les données de passage
    
    La boucle continue jusqu'à :
    - La fin de la vidéo
    - Une interruption utilisateur
    - Une erreur critique
    
    Notes:
        Utilise un système de gestion d'erreurs pour assurer
        une fermeture propre en cas de problème.
    """
    cap, mask, tracker_state = initialize()

    try:
        while True:
            # Lecture et traitement de la frame
            ret, frame = cap.read()
            if not ret:
                break
                
            # Traitement de la frame et mise à jour du tracker
            frame = process_frame(frame, cache_file, detect_squares)
            tracked_persons = update_tracker(tracker_state, frame)
            
            # Traitement des traversées de ligne
            ids_to_process = []
            for person in tracked_persons:
                person_id = person['id']
                if person['value'] is not None:
                    update_color(person_id, person['value'])  # Appel direct de la fonction
                
                if check_line_crossing(person, line_start, line_end):
                    ids_to_process.append(person_id)
            
            # Mise à jour de l'affichage
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
            
            # Vérification de la condition de sortie
            should_quit, elapsed_time = show_frame(frame)
            if should_quit:
                break
                
    except Exception as e:
        print(f"Erreur dans la boucle principale : {e}")
        
    finally:
        print("Fermeture du programme...")
        cleanup()
        cap.release()
        release()

# Point d'entrée du programme
if __name__ == "__main__":
    main() 