import cv2
import numpy as np
import json
import os

class ColorCalibrator:
    """
    Gère la calibration des couleurs et la génération des plages HSV.
    """
    def __init__(self):
        self.samples = {}
        self.calibrated_ranges = {}
        
    def add_sample(self, color_name, frame, roi):
        """
        Ajoute un échantillon pour une couleur donnée.
        
        Args:
            color_name (str): Nom de la couleur
            frame (np.array): Image source
            roi (tuple): Rectangle de sélection (x, y, w, h)
        """
        if color_name not in self.samples:
            self.samples[color_name] = []
            
        # Extraction de la ROI et conversion en HSV
        x, y, w, h = roi
        sample = frame[y:y+h, x:x+w]
        hsv_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
        self.samples[color_name].append(hsv_sample)
        
    def calibrate_color(self, color_name):
        """
        Calibre les plages HSV pour une couleur donnée.
        """
        if not self.samples.get(color_name):
            return None
            
        all_hsv = np.vstack([sample.reshape(-1, 3) 
                            for sample in self.samples[color_name]])
        
        # Calcul des percentiles pour être plus robuste aux outliers
        h_min, h_max = np.percentile(all_hsv[:, 0], [5, 95])
        s_min, s_max = np.percentile(all_hsv[:, 1], [5, 95])
        v_min, v_max = np.percentile(all_hsv[:, 2], [5, 95])
        
        # Ajout de marges
        h_margin, s_margin, v_margin = 10, 30, 30
        
        range_values = {
            "h_min": max(0, h_min - h_margin),
            "h_max": min(180, h_max + h_margin),
            "s_min": max(0, s_min - s_margin),
            "s_max": min(255, s_max + s_margin),
            "v_min": max(0, v_min - v_margin),
            "v_max": min(255, v_max + v_margin)
        }
        
        self.calibrated_ranges[color_name] = range_values
        return range_values
        
    def save_calibration(self, output_file):
        """
        Sauvegarde les plages calibrées dans un fichier JSON.
        Met à jour les calibrations existantes sans écraser les autres.
        """
        # Charger les calibrations existantes
        existing_calibrations = {}
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    existing_calibrations = json.load(f)
            except json.JSONDecodeError:
                print("Attention: Le fichier de calibration existant est corrompu. Création d'un nouveau fichier.")

        # Mettre à jour avec les nouvelles calibrations
        existing_calibrations.update(self.calibrated_ranges)

        # Sauvegarder le tout
        with open(output_file, 'w') as f:
            json.dump(existing_calibrations, f, indent=4)
        
        # Afficher un résumé des couleurs calibrées
        print("\nRésumé des calibrations:")
        for color in self.calibrated_ranges:
            print(f"- {color} : calibré")
        for color in existing_calibrations:
            if color not in self.calibrated_ranges:
                print(f"- {color} : conservé de la calibration précédente") 