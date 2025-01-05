import numpy as np
from collections import defaultdict
from config import MAX_DISAPPEAR_FRAMES, MIN_IOU_THRESHOLD, MIN_CONFIDENCE, IOU_THRESHOLD, modele_path, bytetrack_path
from ultralytics import YOLO
import cv2

class TrackedPerson:
    """
    Représentation d'un coureur suivi dans la vidéo.
    
    Cette classe maintient l'état et les attributs d'un coureur détecté,
    incluant sa position, sa trajectoire et son état de franchissement de ligne.

    Attributes:
        bbox (list): Coordonnées de la boîte englobante [x1, y1, x2, y2]
        id (int): Identifiant unique du coureur
        confidence (float): Score de confiance de la détection
        value (str): Valeur dominante du t-shirt
        disappeared (int): Nombre de frames depuis la dernière détection
        trajectory (list): Liste des positions centrales précédentes
        crossed_line (bool): Indicateur de franchissement de ligne
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
        self.value = None  # Valeur dominante du t-shirt
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
        x1, y1, x2, y2 = map(int, self.bbox)  # Conversion en entiers
        bottom_center_x = x1 + (x2 - x1) // 2  # Point milieu en x
        bottom_center_y = y2  # Point le plus bas de la bbox
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
    Gestionnaire principal du suivi des coureurs.
    
    Utilise ByteTrack pour le suivi des coureurs et gère leur état
    tout au long de la vidéo.

    Attributes:
        next_id (int): Prochain ID disponible pour un nouveau coureur
        persons (dict): Dictionnaire des coureurs actuellement suivis
        counter (defaultdict): Compteur par couleur de t-shirt
        model (YOLO): Modèle YOLO pour la détection
        crossed_ids (set): Ensemble des IDs ayant franchi la ligne

    Notes:
        - Utilise l'algorithme ByteTrack pour un suivi robuste
        - Maintient un état des coureurs ayant déjà franchi la ligne
    """
    def __init__(self):
        self.next_id = 1  # On commence à 1
        self.persons = {}  # Dictionnaire des personnes suivies
        self.counter = defaultdict(int)
        self.model = YOLO(modele_path)
        self.crossed_ids = set()
        self.id_mapping = {}  # Nouveau: mapping entre les IDs de ByteTrack et nos IDs séquentiels

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
            bytetrack_ids = results[0].boxes.id.cpu().numpy()
            
            current_ids = set()
            
            for box, bytetrack_id in zip(boxes, bytetrack_ids):
                bytetrack_id = int(bytetrack_id)
                
                # Si l'ID ByteTrack n'est pas dans notre mapping, lui assigner un nouvel ID séquentiel
                if bytetrack_id not in self.id_mapping:
                    self.id_mapping[bytetrack_id] = self.next_id
                    self.next_id += 1
                
                # Utiliser notre ID séquentiel
                sequential_id = self.id_mapping[bytetrack_id]
                
                # Ignorer les IDs qui ont déjà traversé la ligne
                if sequential_id in self.crossed_ids:
                    continue
                    
                current_ids.add(sequential_id)
                
                if sequential_id not in self.persons:
                    self.persons[sequential_id] = TrackedPerson(box, sequential_id, 1.0)
                else:
                    self.persons[sequential_id].update_position(box, frame)
                    self.persons[sequential_id].disappeared = 0

            # Supprimer les personnes qui ne sont plus détectées
            for person_id in list(self.persons.keys()):
                if person_id not in current_ids:
                    self.persons[person_id].disappeared += 1
                    if self.persons[person_id].disappeared > MAX_DISAPPEAR_FRAMES:
                        del self.persons[person_id]

        return list(self.persons.values())

    def mark_as_crossed(self, person_id):
        """Marque un ID comme ayant traversé la ligne"""
        self.crossed_ids.add(person_id)
        if person_id in self.persons:
            del self.persons[person_id] 