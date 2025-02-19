import signal
import sys
import torch
from config import (
    VIDEO_INPUT_PATH,
    DETECT_SQUARES,
    line_start,
    line_end
)
from video_processor import (
    load_mask,
    setup_video_capture,
    process_frame,
    initialize_color_masks
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
import cProfile



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
    """
    try:
        if not torch.cuda.is_available():
            print("CUDA n'est pas disponible")
            print(f"Version PyTorch: {torch.__version__}")
            return torch.device("cpu")
        
        # Vérification plus détaillée du GPU
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            print("Aucun GPU détecté")
            return torch.device("cpu")
            
        # Sélection du premier GPU disponible
        compute_device = torch.device("cuda:0")
        print(f"GPU détectée: {torch.cuda.get_device_name(0)}")
        print(f"Nombre de GPUs: {gpu_count}")
        print(f"Version CUDA: {torch.cuda.get_device_capability(0)}")
        
        # Test rapide pour vérifier que le GPU fonctionne
        test_tensor = torch.tensor([1.0], device=compute_device)
        if test_tensor.device.type == "cuda":
            print("Test GPU réussi")
        
        return compute_device
        
    except Exception as e:
        print(f"Erreur lors de la configuration du GPU: {str(e)}")
        return torch.device("cpu")

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
    """
    try:
        # Initialisation du gestionnaire de signaux
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        init_detection_history()

        # Initialisation des masques de couleurs (doit être fait avant toute détection)
        initialize_color_masks()
        print("Masques de couleurs initialisés avec succès")
        
        # Configuration de la capture vidéo
        video_capture = setup_video_capture(VIDEO_INPUT_PATH)
        
        # Chargement et pré-calcul du masque de détection
        load_mask()
        print("Masque de détection chargé et pré-calculé")

        tracker_state = create_tracker()
        init_display()  # Initialisation de l'affichage

        compute_device = setup_device()
        tracker_state['person_detection_model'] = tracker_state['person_detection_model'].to(compute_device)

        _, initial_frame = video_capture.read()
        if initial_frame is not None:
            try:
                get_average_colors(initial_frame, True)
            except Exception as e:
                print(f"Erreur lors de la détection des couleurs Macbeth: {e}")
        else:
            print("Impossible de lire la première frame de la vidéo.")

        return {
            'video_capture': video_capture,
            'tracker_state': tracker_state
        }
        
    except Exception as e:
        print(f"Erreur lors de l'initialisation du système : {str(e)}")
        sys.exit(1)

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
    system_components = initialize_system()
    video_capture = system_components['video_capture']
    tracker_state = system_components['tracker_state']

    try:
        while True:
            ret, current_frame = video_capture.read()
            if not ret:
                break
                
            processed_frame = process_frame(current_frame, DETECT_SQUARES)
            tracked_persons = update_tracker(tracker_state, processed_frame)
            
            persons_to_process = []
            for tracked_person in tracked_persons:
                person_id = tracked_person['id']
                if tracked_person['value'] is not None:
                    update_detection_value(person_id, tracked_person['value'])
                
                if check_line_crossing(tracked_person, line_start, line_end):
                    persons_to_process.append(person_id)
            
            for tracked_person in tracked_persons:
                draw_person(processed_frame, tracked_person)
            
            draw_crossing_line(processed_frame, line_start, line_end)
            draw_counters(processed_frame, tracker_state['line_crossing_counter'])
            
            # Modification ici pour récupérer l'heure formatée
            should_exit, _, formatted_time = show_frame(processed_frame)
            
            # Traitement des personnes qui ont traversé la ligne
            for person_id in persons_to_process:
                if person_id in tracker_state['active_tracked_persons']:
                    print(f"!!! Ligne traversée par ID={person_id} !!!")
                    
                    detected_value = get_dominant_detection(person_id)
                    update_detection_value(person_id, detected_value)
                    
                    # Utilisation de l'heure formatée pour l'enregistrement
                    record_crossing(person_id, formatted_time)
                    
                    dominant_value = get_dominant_detection(person_id)
                    if dominant_value:
                        tracker_state['line_crossing_counter'][dominant_value] += 1
                    mark_person_as_crossed(tracker_state, person_id)
            
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
    #cProfile.run('main()')
    main()