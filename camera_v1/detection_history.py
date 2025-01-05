from collections import defaultdict
from datetime import datetime
import csv
from pathlib import Path
import os
from config import DETECTION_MODE

class DetectionHistory:
    """
    Gestionnaire d'historique des couleurs détectées pour chaque coureur.
    
    Cette classe maintient un historique des couleurs détectées pour chaque coureur
    et enregistre les données dans un fichier CSV lors du franchissement de la ligne.

    Attributes:
        color_history (defaultdict): Historique des couleurs par ID de coureur
        output_file (Path): Chemin du fichier CSV de sortie
        csv_file (file): Fichier CSV ouvert en mode append
        csv_writer (csv.writer): Objet writer pour l'écriture CSV

    Notes:
        Le fichier CSV est créé avec un buffer minimal pour assurer
        l'enregistrement immédiat des données.
    """

    def __init__(self, output_file="detections.csv"):
        self.color_history = defaultdict(list)  # {id: [liste des couleurs détectées]}
        current_dir = Path(__file__).parent
        self.output_file = current_dir / output_file
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Vérifie si le fichier existe
        file_exists = self.output_file.exists()
        
        # Ouvre le fichier avec buffer minimal
        self.csv_file = open(self.output_file, 'a', newline='', buffering=1)
        self.csv_writer = csv.writer(self.csv_file)
        
        # Écrit l'en-tête seulement si le fichier est nouveau
        if not file_exists:
            self.csv_writer.writerow(['Timestamp', 'Elapsed Time', 'ID', 'Dominant Color'])


    def update_color(self, person_id, value):
        """
        Met à jour l'historique des valeurs pour un coureur.
        """
        if value is not None:  # Vérifiez que la valeur n'est pas None
            self.color_history[person_id].append(value)

    def get_dominant_value(self, person_id):
        """Retourne la valeur la plus fréquente pour un ID"""
        if not self.color_history[person_id]:
            return None
        
        # Compte les occurrences de chaque valeur
        value_counts = defaultdict(int)
        for value in self.color_history[person_id]:
            if value is not None:  # Ignore les valeurs None
                value_counts[value] += 1
            
        if not value_counts:  # Si aucune valeur valide
            return None
        
        # Retourne la valeur la plus fréquente
        return max(value_counts.items(), key=lambda x: x[1])[0]

    def record_crossing(self, person_id, elapsed_time):
        """Enregistre le passage avec la couleur dominante et le temps écoulé"""
        dominant_value = self.get_dominant_value(person_id)
        if dominant_value:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            elapsed_str = f"{minutes:02d}:{seconds:02d}"
            
            try:
                self.csv_writer.writerow([timestamp, elapsed_str, person_id, dominant_value])
                self.csv_file.flush()
                os.fsync(self.csv_file.fileno())
                print(f"Écriture CSV réussie : {timestamp}, {elapsed_str}, {person_id}, {dominant_value}")  # debug
            except Exception as e:
                print(f"Erreur d'écriture CSV : {e}")
        
        # Nettoyage de l'historique pour cet ID
        if person_id in self.color_history:
            del self.color_history[person_id]

    def __del__(self):
        """Ferme proprement le fichier CSV"""
        self.csv_file.close() 