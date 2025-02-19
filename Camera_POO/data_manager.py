import csv
import json
import os
from datetime import datetime
from config import config

class DataManager:
    """Classe gérant la sauvegarde et le chargement des données"""
    def __init__(self):
        self.detections_data = []
        self.cache_data = {}
        self._load_cache()

    def _load_cache(self):
        """Charge les données du cache si elles existent"""
        if os.path.exists(config.CACHE_FILE_PATH):
            try:
                with open(config.CACHE_FILE_PATH, 'r') as f:
                    self.cache_data = json.load(f)
            except json.JSONDecodeError:
                print("Erreur lors du chargement du cache. Création d'un nouveau cache.")
                self.cache_data = {}

    def save_cache(self):
        """Sauvegarde les données dans le cache"""
        with open(config.CACHE_FILE_PATH, 'w') as f:
            json.dump(self.cache_data, f)

    def add_detection(self, frame_number, tracked_objects):
        """
        Ajoute les détections du frame actuel aux données
        
        Args:
            frame_number: Numéro du frame
            tracked_objects: Dictionnaire des objets suivis
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for obj_id, obj in tracked_objects.items():
            if obj.centroids:  # Vérifier si l'objet a des centroids
                centroid = obj.centroids[-1]
                detection_data = {
                    'timestamp': timestamp,
                    'frame': frame_number,
                    'object_id': obj_id,
                    'x': centroid[0],
                    'y': centroid[1],
                    'color': obj.color if obj.color else "unknown",
                    'direction': obj.direction if obj.direction else "none",
                    'crossed_line': obj.crossed_line
                }
                self.detections_data.append(detection_data)

    def export_to_csv(self):
        """Exporte les données de détection vers un fichier CSV"""
        if not self.detections_data:
            print("Aucune donnée à exporter")
            return

        fieldnames = [
            'timestamp', 'frame', 'object_id', 
            'x', 'y', 'color', 'direction', 'crossed_line'
        ]

        try:
            with open(config.CSV_OUTPUT_PATH, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.detections_data)
            print(f"Données exportées vers {config.CSV_OUTPUT_PATH}")
        except Exception as e:
            print(f"Erreur lors de l'export CSV: {e}")

    def get_statistics(self):
        """
        Calcule les statistiques sur les détections
        
        Returns:
            dict: Statistiques calculées
        """
        stats = {
            'total_detections': len(self.detections_data),
            'unique_objects': len(set(d['object_id'] for d in self.detections_data)),
            'color_distribution': {},
            'direction_counts': {
                'up': 0,
                'down': 0,
                'none': 0
            }
        }

        # Calculer la distribution des couleurs
        for detection in self.detections_data:
            color = detection['color']
            if color not in stats['color_distribution']:
                stats['color_distribution'][color] = 0
            stats['color_distribution'][color] += 1

            # Compter les directions
            direction = detection['direction']
            if direction in stats['direction_counts']:
                stats['direction_counts'][direction] += 1

        return stats

    def save_detection_state(self, frame_number, tracked_objects):
        """
        Sauvegarde l'état des détections dans le cache
        
        Args:
            frame_number: Numéro du frame
            tracked_objects: Dictionnaire des objets suivis
        """
        state = {
            'frame_number': frame_number,
            'objects': {}
        }

        for obj_id, obj in tracked_objects.items():
            state['objects'][str(obj_id)] = {
                'centroids': obj.centroids,
                'color': obj.color,
                'direction': obj.direction,
                'crossed_line': obj.crossed_line
            }

        self.cache_data['last_state'] = state
        self.save_cache()

    def load_detection_state(self):
        """
        Charge le dernier état des détections depuis le cache
        
        Returns:
            tuple: (frame_number, état des objets) ou (None, None) si pas de cache
        """
        if 'last_state' not in self.cache_data:
            return None, None

        state = self.cache_data['last_state']
        return state['frame_number'], state['objects']

    def clear_cache(self):
        """Efface les données du cache"""
        self.cache_data = {}
        if os.path.exists(config.CACHE_FILE_PATH):
            os.remove(config.CACHE_FILE_PATH) 