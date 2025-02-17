import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import os
from config import DETECTION_MODE

# Variables globales pour la gestion de l'historique des détections
person_detection_history = defaultdict(list)  # {person_id: [liste des valeurs détectées]}
csv_output_file = None
csv_output_writer = None

def init_detection_history(csv_file_path):
    """
    Initialise l'historique des détections et crée/ouvre le fichier CSV de sortie.
    
    Cette fonction configure le système de journalisation des détections en ouvrant
    un fichier CSV en mode ajout. Le fichier reste ouvert pour optimiser les performances
    d'écriture.
    
    Args:
        csv_file_path (str): Chemin vers le fichier CSV de sortie
                   
    Notes:
        - Utilise les variables globales csv_file et csv_writer
        - Le fichier est ouvert en mode 'append' avec buffering=1 pour une écriture immédiate
    """
    global csv_output_file, csv_output_writer
    
    csv_output_file = open(csv_file_path, 'a', newline='', buffering=1)
    csv_output_writer = csv.writer(csv_output_file)

def update_detection_value(person_id, detected_value):
    """
    Met à jour l'historique des détections pour une personne.
    
    Args:
        person_id (int): Identifiant unique de la personne
        detected_value (str): Valeur détectée (couleur/numéro)
    
    Notes:
        - Les valeurs None sont ignorées
        - L'historique est stocké dans person_detection_history
    """
    if detected_value is not None:
        person_detection_history[person_id].append(detected_value)

def get_dominant_detection(person_id):
    """
    Détermine la valeur la plus fréquente pour une personne donnée.
    
    Analyse l'historique des détections pour une personne et retourne
    la valeur qui apparaît le plus souvent, permettant de filtrer
    les détections erronées.
    
    Args:
        person_id (int): Identifiant unique de la personne
    
    Returns:
        str or None: La valeur la plus fréquente, ou None si aucune détection valide
    """
    person_detections = person_detection_history[person_id]
    if not person_detections:
        return None
    
    detection_frequencies = defaultdict(int)
    for detected_value in person_detections:
        if detected_value is not None:
            detection_frequencies[detected_value] += 1
        
    if not detection_frequencies:
        return None
    
    return max(detection_frequencies.items(), key=lambda x: x[1])[0]

def record_crossing(person_id, current_elapsed_time):
    """
    Enregistre le passage d'une personne avec sa valeur dominante et le temps écoulé.
    
    Cette fonction :
    1. Récupère la valeur dominante de la personne
    2. Formate le timestamp et le temps écoulé
    3. Enregistre les données dans le fichier CSV
    4. Nettoie l'historique de la personne
    
    Args:
        person_id (int): Identifiant unique de la personne
        current_elapsed_time (float): Temps écoulé depuis le début en secondes
            
    Notes:
        - Le format d'enregistrement CSV est: [timestamp, temps_écoulé, id_coureur, couleur]
        - L'historique du coureur est effacé après l'enregistrement
        - Utilise os.fsync pour garantir l'écriture sur le disque
    """
    dominant_detection = get_dominant_detection(person_id)
    if dominant_detection:
        crossing_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        minutes_elapsed = int(current_elapsed_time // 60)
        seconds_elapsed = int(current_elapsed_time % 60)
        formatted_elapsed_time = f"{minutes_elapsed:02d}:{seconds_elapsed:02d}"
        
        try:
            csv_output_writer.writerow([crossing_timestamp, formatted_elapsed_time, person_id, dominant_detection])
            csv_output_file.flush()
            os.fsync(csv_output_file.fileno())
            print(f"Enregistrement CSV réussi : {crossing_timestamp}, {formatted_elapsed_time}, "
                  f"{person_id}, {dominant_detection}")
        except Exception as e:
            print(f"Erreur d'enregistrement CSV : {e}")
    
    if person_id in person_detection_history:
        del person_detection_history[person_id]

def cleanup():
    """
    Ferme proprement le fichier CSV.
    
    Cette fonction doit être appelée à la fin du programme pour assurer
    que toutes les données sont bien écrites et que les ressources
    sont libérées correctement.
    """
    if csv_output_file:
        csv_output_file.close() 