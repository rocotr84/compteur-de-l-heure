import cv2
import os
import numpy as np
from ultralytics import YOLO
from video_processor import VideoProcessor
from display_manager import DisplayManager
from tracker import PersonTracker
from config import *
from color_history import ColorHistory
import signal
import sys

"""
Module principal de l'application de comptage de personnes.
Gère l'initialisation des composants, le traitement vidéo et la coordination
entre les différents modules (détection, tracking, affichage).
"""

# Variable globale pour le color_tracker
color_tracker = None

def signal_handler(sig, frame):
    """Gestionnaire pour l'arrêt propre du programme"""
    print("\nSauvegarde des données et arrêt du programme...")
    if color_tracker:
        # Enregistre les données restantes pour chaque personne suivie
        for person_id in list(color_tracker.color_history.keys()):
            color_tracker.record_crossing(person_id)
        del color_tracker
    sys.exit(0)

def main():
    """
    Fonction principale du programme
    """
    global color_tracker
    
    # Initialisation du gestionnaire de signaux
    signal.signal(signal.SIGINT, signal_handler)  # Pour Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Pour kill
    
    color_tracker = ColorHistory()
    print(f"Démarrage du suivi des couleurs...")
    
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
            
            # Liste des IDs à supprimer après le traitement
            ids_to_remove = []
            
            # Traitement de chaque personne
            for person_id, person in tracker.persons.items():
                # Mise à jour de la couleur
                if hasattr(person, 'color') and person.color is not None:
                    color_tracker.update_color(person_id, person.color)
                
                # Vérification du franchissement de ligne
                if person.check_line_crossing(line_start, line_end):
                    print(f"!!! Ligne traversée par ID={person_id} !!!")
                    # Enregistrement dans le CSV
                    color_tracker.record_crossing(person_id)
                    # Mise à jour du compteur
                    dominant_color = color_tracker.get_dominant_color(person_id)
                    if dominant_color:
                        tracker.counter[dominant_color] += 1
                    # Marquer pour suppression
                    ids_to_remove.append(person_id)
            
            # Suppression des personnes ayant traversé la ligne
            for person_id in ids_to_remove:
                if person_id in tracker.persons:
                    del tracker.persons[person_id]
                    print(f"Personne ID={person_id} supprimée du tracking")
            
            # Affichage
            for person in tracked_persons:
                display.draw_person(frame, person)
            
            display.draw_crossing_line(frame, line_start, line_end)
            display.draw_counters(frame, tracker.counter)
            
            if display.show_frame(frame):
                break
                
    except Exception as e:
        print(f"Erreur dans la boucle principale : {e}")
        
    finally:
        print("Fermeture du programme...")
        if color_tracker:
            # Sauvegarde finale des données
            for person_id in list(color_tracker.color_history.keys()):
                color_tracker.record_crossing(person_id)
        cap.release()
        display.release()

# Point d'entrée du programme
if __name__ == "__main__":
    main() 