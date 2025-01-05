import os
import sys

# Ajout du chemin du projet pour les imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.color_calibrator import ColorCalibrator
from src.gui_manager import CalibrationGUI
from config import SAMPLES_DIR, OUTPUT_DIR, VIDEO_SOURCE

def main():
    # Vérification initiale
    if not os.path.exists(VIDEO_SOURCE):
        print(f"Erreur: Le fichier vidéo {VIDEO_SOURCE} n'existe pas!")
        print(f"Veuillez placer votre vidéo à l'emplacement suivant: {VIDEO_SOURCE}")
        return

    # Création des dossiers nécessaires
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"=== Programme de calibration des couleurs ===")
    print(f"Vidéo source: {VIDEO_SOURCE}")
    
    # Initialisation
    calibrator = ColorCalibrator()
    gui = CalibrationGUI(calibrator)
    
    # Lancement de l'interface
    print("Démarrage de la calibration...")
    gui.run(VIDEO_SOURCE)
    
    # Sauvegarde de la calibration
    output_file = os.path.join(OUTPUT_DIR, "calibration.json")
    calibrator.save_calibration(output_file)
    print(f"Calibration sauvegardée dans {output_file}")

if __name__ == "__main__":
    main() 