import cv2
import argparse
import os
from config import config
from video_processor import VideoProcessor
from data_manager import DataManager

class MacbethTracker:
    """Classe principale de l'application de tracking Macbeth"""
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.data_manager = DataManager()

    def run(self):
        """Lance l'application"""
        print("Démarrage du tracking Macbeth...")
        print(f"Vidéo source: {config.VIDEO_INPUT_PATH}")
        print(f"Mode de détection: {config.DETECTION_MODE}")

        try:
            # Vérifier si une session précédente existe
            frame_number, saved_state = self.data_manager.load_detection_state()
            if frame_number is not None:
                print(f"Reprise de la session précédente au frame {frame_number}")
                # Positionner la vidéo au bon frame
                if self.video_processor.video_capture is not None:
                    self.video_processor.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # Lancer le traitement vidéo
            self.video_processor.run()

        except KeyboardInterrupt:
            print("\nArrêt demandé par l'utilisateur")
        except Exception as e:
            print(f"Erreur lors de l'exécution: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Nettoie les ressources et sauvegarde les données"""
        print("Sauvegarde des données...")
        self.data_manager.export_to_csv()
        
        # Afficher les statistiques finales
        stats = self.data_manager.get_statistics()
        print("\nStatistiques finales:")
        print(f"Total des détections: {stats['total_detections']}")
        print(f"Objets uniques: {stats['unique_objects']}")
        print("\nDistribution des couleurs:")
        for color, count in stats['color_distribution'].items():
            print(f"  {color}: {count}")
        print("\nComptage des directions:")
        for direction, count in stats['direction_counts'].items():
            print(f"  {direction}: {count}")

        # Nettoyage final
        self.video_processor.cleanup()
        print("Application terminée")

def check_resources():
    """Vérifie que toutes les ressources nécessaires sont présentes"""
    required_files = [
        config.VIDEO_INPUT_PATH,
        config.MODEL_PATH,
        config.DETECTION_MASK_PATH
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(f"Ressources manquantes: {', '.join(missing_files)}")

def parse_arguments():
    """Parse les arguments de la ligne de commande"""
    parser = argparse.ArgumentParser(description='Macbeth Tracking System')
    parser.add_argument('--video', type=str, help='Chemin vers le fichier vidéo')
    parser.add_argument('--mode', choices=['color', 'number'], 
                       help='Mode de détection (color ou number)')
    parser.add_argument('--show-gui', action='store_true', default=True,
                       help='Afficher l\'interface graphique')
    parser.add_argument('--save-video', action='store_true', default=False,
                       help='Sauvegarder la vidéo de sortie')
    return parser.parse_args()

def update_config(args):
    """Met à jour la configuration avec les arguments de la ligne de commande"""
    try:
        if args.video and os.path.exists(args.video):
            config.VIDEO_INPUT_PATH = args.video
        if args.mode:
            config.DETECTION_MODE = args.mode
        
        # Mise à jour des paramètres d'affichage
        config.SAVE_VIDEO = bool(args.save_video)
        if not args.show_gui:
            config.SHOW_ROI_AND_COLOR = False
            config.SHOW_TRAJECTORIES = False
            config.SHOW_CENTER = False
            config.SHOW_LABELS = False
    except Exception as e:
        print(f"Erreur lors de la mise à jour de la configuration: {e}")
        raise

if __name__ == "__main__":
    try:
        args = parse_arguments()
        update_config(args)
        check_resources()
        app = MacbethTracker()
        app.run()
    except FileNotFoundError as e:
        print(f"Erreur: {e}")
    except Exception as e:
        print(f"Erreur critique: {e}")