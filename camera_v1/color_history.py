from collections import defaultdict
from datetime import datetime
import csv
from pathlib import Path
import os

class ColorHistory:
    def __init__(self, output_file="detections.csv"):
        self.color_history = defaultdict(list)  # {id: [liste des couleurs détectées]}
        current_dir = Path(__file__).parent
        self.output_file = current_dir / output_file
        print(f"Fichier CSV créé à : {self.output_file}")  # Debug
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Vérifie si le fichier existe
        file_exists = self.output_file.exists()
        
        # Ouvre le fichier avec buffer minimal
        self.csv_file = open(self.output_file, 'a', newline='', buffering=1)
        self.csv_writer = csv.writer(self.csv_file)
        
        # Écrit l'en-tête seulement si le fichier est nouveau
        if not file_exists:
            self.csv_writer.writerow(['Timestamp', 'ID', 'Dominant Color'])
            print(f"Fichier CSV créé : {self.output_file}")

    def update_color(self, person_id, color):
        """Ajoute une couleur détectée pour un ID"""
        self.color_history[person_id].append(color)

    def get_dominant_color(self, person_id):
        """Retourne la couleur la plus fréquente pour un ID"""
        if not self.color_history[person_id]:
            return None
        
        # Compte les occurrences de chaque couleur
        color_counts = defaultdict(int)
        for color in self.color_history[person_id]:
            color_counts[color] += 1
            
        # Retourne la couleur la plus fréquente
        return max(color_counts.items(), key=lambda x: x[1])[0]

    def record_crossing(self, person_id):
        """Enregistre le passage avec la couleur dominante"""
        dominant_color = self.get_dominant_color(person_id)
        if dominant_color:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                print(f"Tentative d'écriture dans le CSV pour ID={person_id}")  # Debug
                self.csv_writer.writerow([timestamp, person_id, dominant_color])
                self.csv_file.flush()
                os.fsync(self.csv_file.fileno())
                print(f"Écriture CSV réussie : {timestamp}, {person_id}, {dominant_color}")  # Debug
            except Exception as e:
                print(f"Erreur d'écriture CSV : {e}")
            
        # Nettoyage de l'historique pour cet ID
        if person_id in self.color_history:
            del self.color_history[person_id]

    def __del__(self):
        """Ferme proprement le fichier CSV"""
        self.csv_file.close() 