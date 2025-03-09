"""
Module principal de l'application.

Ce module définit la classe Application qui orchestre tous les composants
du système de détection et de suivi.
"""

import signal
import sys
import torch
import time
from datetime import datetime

from config.paths_config import VIDEO_INPUT_PATH
from config.detection_config import DETECT_SQUARES
from config.display_config import line_start, line_end

from src.video_processor import (
    load_mask,
    setup_video_capture,
    process_frame,
    initialize_color_masks
) 
from src.display_manager import (
    init_display,
    draw_person,
    draw_counters,
    draw_crossing_line,
    show_frame,
    release_display
)
from src.tracker import (
    create_tracker,
    update_tracker,
    check_line_crossing,
    mark_person_as_crossed
)
from src.detection_history import (
    cleanup,
    init_detection_history,
    update_detection_value,
    get_dominant_detection,
    record_crossing
)
from src.macbeth_color_and_rectangle_detector import get_average_colors

class Application:
    """
    Classe principale de l'application de comptage de personnes.
    
    Cette classe coordonne les différentes composantes du système:
    - Initialisation des ressources
    - Traitement vidéo
    - Détection et suivi
    - Affichage et enregistrement
    - Gestion des événements
    
    Attributes:
        video_capture: Objet de capture vidéo
        tracker_state: État du tracker de personnes
        running (bool): Indique si l'application est en cours d'exécution
    """
    
    def __init__(self):
        """Initialise l'application et ses composants."""
        self.video_capture = None
        self.tracker_state = None
        self.running = False
        
        # Configuration des gestionnaires de signaux
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signal_received, frame):
        """
        Gestionnaire pour l'arrêt propre du programme.
        
        Args:
            signal_received: Signal reçu
            frame: Frame d'exécution courante
        """
        print("\nSauvegarde des données et arrêt du programme...")
        self.cleanup()
        sys.exit(0)
    
    def setup_device(self):
        """
        Configure le dispositif de calcul (GPU/CPU) pour le traitement.
        
        Returns:
            torch.device: Dispositif à utiliser pour les calculs
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
    
    def initialize(self):
        """
        Initialise tous les composants nécessaires au fonctionnement du programme.
        
        Cette méthode:
        1. Initialise l'historique de détection
        2. Configure le processeur vidéo et charge le masque
        3. Initialise le tracker et l'affichage
        4. Configure le dispositif de calcul
        5. Effectue la détection initiale des couleurs Macbeth
        
        Returns:
            bool: True si l'initialisation a réussi, False sinon
        """
        try:
            init_detection_history()

            # Initialisation des masques de couleurs (doit être fait avant toute détection)
            initialize_color_masks()
            print("Masques de couleurs initialisés avec succès")
            
            # Configuration de la capture vidéo
            self.video_capture = setup_video_capture(VIDEO_INPUT_PATH)
            
            # Chargement et pré-calcul du masque de détection
            load_mask()
            print("Masque de détection chargé et pré-calculé")

            self.tracker_state = create_tracker()
            init_display()  # Initialisation de l'affichage

            compute_device = self.setup_device()
            self.tracker_state['person_detection_model'] = self.tracker_state['person_detection_model'].to(compute_device)

            _, initial_frame = self.video_capture.read()
            if initial_frame is not None:
                try:
                    get_average_colors(initial_frame, True)
                except Exception as e:
                    print(f"Erreur lors de la détection des couleurs Macbeth: {e}")
            else:
                print("Impossible de lire la première frame de la vidéo.")
                return False

            self.running = True
            return True
            
        except Exception as e:
            print(f"Erreur lors de l'initialisation du système : {str(e)}")
            return False
    
    def process_frame_with_tracking(self, current_frame):
        """
        Traite une frame avec détection et suivi des personnes.
        
        Args:
            current_frame (np.ndarray): Frame brute à traiter
            
        Returns:
            tuple: (processed_frame, should_exit, formatted_time)
        """
        processed_frame = process_frame(current_frame, DETECT_SQUARES)
        tracked_persons = update_tracker(self.tracker_state, processed_frame)
        
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
        draw_counters(processed_frame, self.tracker_state['line_crossing_counter'])
        
        # Affichage de la frame
        should_exit, _, formatted_time = show_frame(processed_frame)
        
        # Traitement des personnes qui ont traversé la ligne
        for person_id in persons_to_process:
            if person_id in self.tracker_state['active_tracked_persons']:
                print(f"!!! Ligne traversée par ID={person_id} !!!")
                
                detected_value = get_dominant_detection(person_id)
                update_detection_value(person_id, detected_value)
                
                # Utilisation de l'heure formatée pour l'enregistrement
                record_crossing(person_id, formatted_time)
                
                dominant_value = get_dominant_detection(person_id)
                if dominant_value:
                    self.tracker_state['line_crossing_counter'][dominant_value] += 1
                mark_person_as_crossed(self.tracker_state, person_id)
        
        return processed_frame, should_exit, formatted_time
    
    def run(self):
        """
        Exécute la boucle principale de l'application.
        
        Cette méthode gère la boucle principale qui:
        1. Lit les frames de la vidéo
        2. Traite chaque frame (correction couleur, détection)
        3. Met à jour le suivi des personnes
        4. Détecte les traversées de ligne
        5. Met à jour les compteurs et l'affichage
        6. Enregistre les données de passage
        """
        if not self.running:
            if not self.initialize():
                print("Échec de l'initialisation, impossible de démarrer l'application")
                return
        
        try:
            while True:
                ret, current_frame = self.video_capture.read()
                if not ret:
                    break
                    
                _, should_exit, _ = self.process_frame_with_tracking(current_frame)
                
                if should_exit:
                    break
                    
        except Exception as e:
            print(f"Erreur dans la boucle principale : {e}")
            
        finally:
            self.cleanup()
    
    def cleanup(self):
        """
        Nettoie les ressources utilisées par l'application.
        
        Cette méthode assure une fermeture propre:
        - Sauvegarde des données
        - Fermeture des fichiers
        - Libération des ressources vidéo
        """
        print("Fermeture de l'application...")
        cleanup()  # Nettoyage de l'historique de détection
        
        if self.video_capture is not None:
            self.video_capture.release()
            
        release_display()
        self.running = False