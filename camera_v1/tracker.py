import numpy as np
from collections import defaultdict
from config import MAX_DISAPPEAR_FRAMES, MIN_IOU_THRESHOLD

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
        """Calcule le point central de la boîte englobante"""
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
        
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
            
        p1 = self.trajectory[-2]  # Position précédente
        p2 = self.trajectory[-1]  # Position actuelle
        
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
        
    def calculate_iou(self, bbox1, bbox2):
        """
        Calcule l'Intersection over Union entre deux boîtes englobantes
        Args:
            bbox1, bbox2 (list): Coordonnées des boîtes [x1, y1, x2, y2]
        Returns:
            float: Score IoU entre 0 et 1
        """
        # Calcul de l'intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        return intersection / float(area1 + area2 - intersection)
        
    def update(self, detections, frame, confidences):
        """
        Met à jour l'état de toutes les personnes suivies
        Args:
            detections (list): Liste des nouvelles détections
            frame (np.array): Image courante
            confidences (list): Scores de confiance des détections
        Returns:
            list: Liste des personnes actuellement suivies
        """
        # 1. Mise à jour des personnes disparues
        for person_id in list(self.persons.keys()):
            self.persons[person_id].disappeared += 1
            if self.persons[person_id].disappeared > MAX_DISAPPEAR_FRAMES:
                del self.persons[person_id]
                
        # 2. Traitement des nouvelles détections
        if len(detections) > 0:
            # Cas simple : aucune personne suivie
            if len(self.persons) == 0:
                for i, bbox in enumerate(detections):
                    self.persons[self.next_id] = TrackedPerson(bbox, self.next_id, confidences[i])
                    self.next_id += 1
            else:
                # Association des détections aux personnes existantes
                person_ids = list(self.persons.keys())
                person_bboxes = [self.persons[id].bbox for id in person_ids]
                
                # Calcul de la matrice IoU
                iou_matrix = np.zeros((len(detections), len(person_ids)))
                for i, detection in enumerate(detections):
                    for j, person_bbox in enumerate(person_bboxes):
                        iou_matrix[i, j] = self.calculate_iou(detection, person_bbox)
                
                # Association basée sur le meilleur score IoU
                used_detections = set()
                used_persons = set()
                
                while True:
                    if np.max(iou_matrix) < MIN_IOU_THRESHOLD:
                        break
                        
                    i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                    if i not in used_detections and j not in used_persons:
                        person_id = person_ids[j]
                        self.persons[person_id].update_position(detections[i], frame)
                        self.persons[person_id].disappeared = 0
                        used_detections.add(i)
                        used_persons.add(j)
                    iou_matrix[i, j] = 0
                
                # Création de nouvelles personnes pour les détections non associées
                for i, bbox in enumerate(detections):
                    if i not in used_detections:
                        self.persons[self.next_id] = TrackedPerson(bbox, self.next_id, confidences[i])
                        self.next_id += 1
                        
        return list(self.persons.values()) 