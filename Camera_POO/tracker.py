from collections import OrderedDict
import numpy as np
from config import config
from object_detector import ObjectDetector
from color_detector import ColorDetector

class TrackedObject:
    """Classe représentant un objet suivi"""
    def __init__(self, object_id, centroid, color=None, number=None):
        self.object_id = object_id
        self.centroids = [centroid]
        self.color = color
        self.number = number
        self.disappeared = 0
        self.crossed_line = False
        self.direction = None

class ObjectTracker:
    """Classe gérant le suivi des objets détectés"""
    def __init__(self):
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.next_object_id = 0
        self.object_detector = ObjectDetector()
        self.color_detector = ColorDetector()
        
    def register(self, centroid, color=None, number=None):
        """Enregistre un nouvel objet"""
        self.objects[self.next_object_id] = TrackedObject(
            self.next_object_id, 
            centroid, 
            color=color, 
            number=number
        )
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """Supprime un objet du suivi"""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def get_centroid(self, bbox):
        """Calcule le centroïde d'une boîte de détection"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def update(self, detections):
        """
        Met à jour le suivi des objets
        
        Args:
            detections: Liste des détections du frame actuel
            
        Returns:
            OrderedDict: Dictionnaire des objets suivis
        """
        # Si aucune détection, incrémenter les compteurs de disparition
        if len(detections) == 0:
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id].disappeared += 1
                if self.objects[obj_id].disappeared > config.MAX_DISAPPEAR_FRAMES:
                    self.deregister(obj_id)
            return self.objects

        # Initialiser la matrice des centroids actuels
        current_centroids = np.zeros((len(detections), 2), dtype="int")
        for (i, det) in enumerate(detections):
            current_centroids[i] = self.get_centroid(det['bbox'])

        # Si nous ne suivons aucun objet, les enregistrer tous
        if len(self.objects) == 0:
            for i in range(len(current_centroids)):
                self.register(current_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            previous_centroids = np.array([obj.centroids[-1] 
                                         for obj in self.objects.values()])

            # Calculer les distances entre les anciens et nouveaux centroids
            D = np.linalg.norm(previous_centroids[:, np.newaxis] - current_centroids, axis=2)
            
            # Trouver les meilleures correspondances
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id].centroids.append(current_centroids[col])
                self.objects[object_id].disappeared = 0

                used_rows.add(row)
                used_cols.add(col)

            # Gérer les objets non appariés
            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.objects[object_id].disappeared += 1
                    if self.objects[object_id].disappeared > config.MAX_DISAPPEAR_FRAMES:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(current_centroids[col])

        return self.objects

    def check_line_crossing(self, line_start, line_end):
        """Vérifie si les objets ont traversé la ligne de comptage"""
        for obj in self.objects.values():
            if len(obj.centroids) < 2 or obj.crossed_line:
                continue

            p1 = np.array(obj.centroids[-2])
            p2 = np.array(obj.centroids[-1])
            p3 = np.array(line_start)
            p4 = np.array(line_end)

            # Vérifier l'intersection
            if self._segments_intersect(p1, p2, p3, p4):
                obj.crossed_line = True
                # Déterminer la direction
                obj.direction = "up" if p2[1] < p1[1] else "down"

    def _segments_intersect(self, p1, p2, p3, p4):
        """Vérifie si deux segments se croisent"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def analyze_object(self, frame, detection):
        """Analyse la couleur et/ou le numéro d'un objet détecté"""
        roi, _ = self.object_detector.get_roi(frame, detection['bbox'])
        color, confidence = self.color_detector.get_dominant_color(roi)
        return color if confidence > config.MIN_COLOR_WEIGHT else None 