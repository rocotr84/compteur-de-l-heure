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
    """Initialise l'historique des détections et le fichier CSV"""
    global csv_file, csv_writer
    
    csv_file = open(output_file, 'a', newline='', buffering=1)
    csv_writer = csv.writer(csv_file)


def update_color(person_id, value):
    """Met à jour l'historique des valeurs pour un coureur."""
    if value is not None:
        color_history[person_id].append(value)

def get_dominant_value(person_id):
    """Retourne la valeur la plus fréquente pour un ID"""
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
    """Enregistre le passage avec la couleur dominante et le temps écoulé"""
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
    """Ferme proprement le fichier CSV"""
    if csv_file:
        csv_file.close() 