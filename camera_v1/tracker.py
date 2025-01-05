import numpy as np
from collections import defaultdict
from config import MAX_DISAPPEAR_FRAMES, MIN_IOU_THRESHOLD, MIN_CONFIDENCE, IOU_THRESHOLD, modele_path, bytetrack_path
from ultralytics import YOLO
import cv2

class TrackedPerson:
    """
    Classe représentant une personne suivie dans la vidéo.
    Gère la position, la trajectoire et l'état de la personne.
    """
    def __init__(self, bbox, id, confidence):
        """
        Initialise une nouvelle personne suivie
        Args:
            bbox (list): Coordonnées de la boîte englobante [x1, y1, x2, y2]
            id (int): Identifiant unique de la personne
            confidence (float): Score de confiance de la détection
        """
        self.bbox = bbox
        self.id = id
        self.confidence = confidence
        self.color = None  # Couleur dominante du t-shirt
        self.disappeared = 0  # Nombre de frames depuis la dernière détection
        self.trajectory = []  # Liste des positions centrales précédentes
        self.crossed_line = False  # Indique si la personne a déjà traversé la ligne
        
    def update_position(self, bbox, frame=None):
        """
        Met à jour la position de la personne et sa trajectoire
        Args:
            bbox (list): Nouvelles coordonnées [x1, y1, x2, y2]
            frame (np.array, optional): Image pour l'analyse de couleur
        """
        self.bbox = bbox
        center = self.get_center()
        self.trajectory.append(center)
        # Limite la longueur de la trajectoire pour éviter une utilisation excessive de mémoire
        if len(self.trajectory) > 30:
            self.trajectory.pop(0)
            
    def get_center(self):
        """Calcule le point milieu du bas de la bbox de la personne"""
        x, y, w, h = self.bbox
        bottom_center_x = x + w // 2
        bottom_center_y = y + h  # Point le plus bas de la bbox
        return (bottom_center_x, bottom_center_y)
        
    def check_line_crossing(self, line_start, line_end):
        """
        Vérifie si la personne traverse la ligne de comptage
        Args:
            line_start (tuple): Point de début de la ligne (x, y)
            line_end (tuple): Point de fin de la ligne (x, y)
        Returns:
            bool: True si la ligne est traversée pour la première fois
        """
        if len(self.trajectory) < 2 or self.crossed_line:
            return False
            
        p1 = self.trajectory[-2]  # Avant-dernière position
        p2 = self.trajectory[-1]  # Dernière position
        
        def ccw(A, B, C):
            """Test d'orientation pour détecter l'intersection de segments"""
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
            
        # Vérifie si les segments se croisent
        intersect = ccw(p1, line_start, line_end) != ccw(p2, line_start, line_end) and \
                   ccw(p1, p2, line_start) != ccw(p1, p2, line_end)
                   
        if intersect:
            self.crossed_line = True
            return True
        return False

class PersonTracker:
    """
    Gère le suivi de plusieurs personnes dans la vidéo.
    Utilise l'IoU (Intersection over Union) pour associer les détections.
    """
    def __init__(self):
        self.next_id = 1  # Prochain ID disponible
        self.persons = {}  # Dictionnaire des personnes suivies
        self.counter = defaultdict(int)  # Compteur par couleur
        self.model = YOLO(modele_path)  # Ajout du modèle YOLO
        
    def update(self, frame, confidences=None):
        """
        Met à jour l'état de toutes les personnes suivies en utilisant ByteTrack
        Args:
            frame (np.array): Image courante
            confidences: non utilisé, gardé pour compatibilité
        Returns:
            list: Liste des personnes actuellement suivies
        """
        # Détection et tracking avec ByteTrack
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=bytetrack_path,
            classes=0,
            conf=MIN_CONFIDENCE,
            iou=IOU_THRESHOLD,
            verbose=False
        )

        # Mise à jour des personnes suivies
        if results and len(results) > 0 and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            
            # Mettre à jour ou créer les personnes suivies
            current_ids = set()
            for box, track_id in zip(boxes, ids):
                current_ids.add(int(track_id))
                if track_id not in self.persons:
                    self.persons[track_id] = TrackedPerson(box, track_id, 1.0)  # Confiance fixée à 1.0
                else:
                    self.persons[track_id].update_position(box, frame)
                    self.persons[track_id].disappeared = 0

            # Supprimer les personnes qui ne sont plus détectées
            for person_id in list(self.persons.keys()):
                if person_id not in current_ids:
                    self.persons[person_id].disappeared += 1
                    if self.persons[person_id].disappeared > MAX_DISAPPEAR_FRAMES:
                        del self.persons[person_id]

        return list(self.persons.values()) 