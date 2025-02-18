import signal
import sys
import torch
from config import (
    VIDEO_INPUT_PATH,
    DETECTION_MASK_PATH,
    CSV_OUTPUT_PATH,
    CACHE_FILE_PATH,
    DETECT_SQUARES,
    DETECTION_MODE,
    line_start,
    line_end
)
from video_processor import (
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
    release_display
)
from tracker import (
    create_tracker,
    update_tracker,
    check_line_crossing,
    mark_person_as_crossed
)
from detection_history import (
    cleanup,
    init_detection_history,
    update_detection_value,
    get_dominant_detection,
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


def signal_handler(signal_received, frame):
    """
    Gestionnaire pour l'arrêt propre du programme.
    
    Assure la sauvegarde des données et la fermeture correcte des ressources
    lors de l'interruption du programme (Ctrl+C ou SIGTERM).
    
    Args:
        signal_received: Signal reçu
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
        compute_device = torch.device("cuda")
        print(f"GPU détectée: {torch.cuda.get_device_name()}")
    else:
        compute_device = torch.device("cpu")
        print("GPU non disponible, utilisation du CPU")
    return compute_device

def initialize_system():
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
        tuple: (video_capture, detection_mask, tracker_state)
    """
    # Initialisation du gestionnaire de signaux
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    init_detection_history(CSV_OUTPUT_PATH)
    print(f"Démarrage du suivi en mode {DETECTION_MODE}...")

    detection_mask = load_mask(DETECTION_MASK_PATH)
    video_capture = setup_video_capture(VIDEO_INPUT_PATH)

    tracker_state = create_tracker()
    init_display()  # Initialisation de l'affichage

    compute_device = setup_device()
    tracker_state['person_detection_model'] = tracker_state['person_detection_model'].to(compute_device)

    _, initial_frame = video_capture.read()
    if initial_frame is not None:
        get_average_colors(initial_frame, CACHE_FILE_PATH, True)
    else:
        print("Impossible de lire la première frame de la vidéo.")

    return video_capture, detection_mask, tracker_state

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
    video_capture, detection_mask, tracker_state = initialize_system()

    try:
        while True:
            ret, current_frame = video_capture.read()
            if not ret:
                break
                
            processed_frame = process_frame(current_frame, CACHE_FILE_PATH, DETECT_SQUARES)
            tracked_persons = update_tracker(tracker_state, processed_frame)
            
            persons_to_process = []
            for tracked_person in tracked_persons:
                person_id = tracked_person['id']
                if tracked_person['value'] is not None:
                    update_detection_value(person_id, tracked_person['value'])
                
                if check_line_crossing(tracked_person, line_start, line_end):
                    persons_to_process.append(person_id)
            
            for person_id in persons_to_process:
                if person_id in tracker_state['active_tracked_persons']:
                    print(f"!!! Ligne traversée par ID={person_id} !!!")
                    current_elapsed_time = draw_timer(processed_frame)
                    
                    detected_value = get_dominant_detection(person_id)
                    update_detection_value(person_id, detected_value)
                    
                    record_crossing(person_id, current_elapsed_time)
                    dominant_value = get_dominant_detection(person_id)
                    if dominant_value:
                        tracker_state['line_crossing_counter'][dominant_value] += 1
                    mark_person_as_crossed(tracker_state, person_id)
                    print(f"Personne ID={person_id} marquée comme ayant traversé")
            
            for tracked_person in tracked_persons:
                draw_person(processed_frame, tracked_person)
            
            draw_crossing_line(processed_frame, line_start, line_end)
            draw_counters(processed_frame, tracker_state['line_crossing_counter'])
            
            should_exit, _ = show_frame(processed_frame)
            if should_exit:
                break
                
    except Exception as e:
        print(f"Erreur dans la boucle principale : {e}")
        
    finally:
        print("Fermeture du programme...")
        cleanup()
        video_capture.release()
        release_display()

# Point d'entrée du programme
if __name__ == "__main__":
    main() 