from collections import defaultdict
from datetime import datetime
import csv
from pathlib import Path
import os
from config import DETECTION_MODE

# Variables globales
color_history = defaultdict(list)  # {id: [liste des couleurs détectées]}
csv_file = None
csv_writer = None

def init_detection_history(output_file):
    """
    Initialise l'historique des détections et crée/ouvre le fichier CSV de sortie.
    
    Cette fonction configure le système de journalisation des détections en ouvrant
    un fichier CSV en mode ajout. Le fichier reste ouvert pour optimiser les performances
    d'écriture.
    
    Args:
        output_file (str): Chemin vers le fichier CSV de sortie
    
    Notes:
        - Utilise les variables globales csv_file et csv_writer
        - Le fichier est ouvert en mode 'append' avec buffering=1 pour une écriture immédiate
    """
    global csv_file, csv_writer
    
    csv_file = open(output_file, 'a', newline='', buffering=1)
    csv_writer = csv.writer(csv_file)


def update_color(person_id, value):
    """
    Met à jour l'historique des valeurs détectées pour un coureur.
    
    Ajoute une nouvelle détection de couleur à l'historique d'un coureur identifié.
    Les valeurs sont stockées dans un dictionnaire global color_history.
    
    Args:
        person_id (int): Identifiant unique du coureur
        value (str): Valeur de couleur détectée (ou None si pas de détection)
    """
    if value is not None:
        color_history[person_id].append(value)

def get_dominant_value(person_id):
    """
    Détermine la valeur la plus fréquente pour un coureur donné.
    
    Analyse l'historique des détections pour un coureur et retourne
    la valeur qui apparaît le plus souvent, permettant de filtrer
    les détections erronées.
    
    Args:
        person_id (int): Identifiant unique du coureur
    
    Returns:
        str or None: La valeur la plus fréquente, ou None si aucune détection valide
    """
    if not color_history[person_id]:
        return None
    
    value_counts = defaultdict(int)
    for value in color_history[person_id]:
        if value is not None:
            value_counts[value] += 1
        
    if not value_counts:
        return None
    
    return max(value_counts.items(), key=lambda x: x[1])[0]

def record_crossing(person_id, elapsed_time):
    """
    Enregistre le passage d'un coureur avec sa couleur dominante et le temps écoulé.
    
    Cette fonction :
    1. Récupère la couleur dominante du coureur
    2. Formate le timestamp et le temps écoulé
    3. Enregistre les données dans le fichier CSV
    4. Nettoie l'historique du coureur
    
    Args:
        person_id (int): Identifiant unique du coureur
        elapsed_time (float): Temps écoulé depuis le début en secondes
    
    Notes:
        - Le format d'enregistrement CSV est: [timestamp, temps_écoulé, id_coureur, couleur]
        - L'historique du coureur est effacé après l'enregistrement
        - Utilise os.fsync pour garantir l'écriture sur le disque
    """
    dominant_value = get_dominant_value(person_id)
    if dominant_value:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        elapsed_str = f"{minutes:02d}:{seconds:02d}"
        
        try:
            csv_writer.writerow([timestamp, elapsed_str, person_id, dominant_value])
            csv_file.flush()
            os.fsync(csv_file.fileno())
            print(f"Écriture CSV réussie : {timestamp}, {elapsed_str}, {person_id}, {dominant_value}")
        except Exception as e:
            print(f"Erreur d'écriture CSV : {e}")
    
    if person_id in color_history:
        del color_history[person_id]

def cleanup():
    """
    Ferme proprement le fichier CSV.
    
    Cette fonction doit être appelée à la fin du programme pour assurer
    que toutes les données sont bien écrites et que les ressources
    sont libérées correctement.
    """
    if csv_file:
        csv_file.close() 