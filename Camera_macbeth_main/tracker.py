import numpy as np
from collections import defaultdict
from config import (
    MAX_DISAPPEAR_FRAMES,
    MIN_CONFIDENCE,
    IOU_THRESHOLD,
    modele_path,
    bytetrack_path
)
from ultralytics import YOLO

def create_tracked_person(bbox, id, confidence):
    """
    Crée un dictionnaire représentant une personne suivie.
    
    Args:
        bbox (tuple): Coordonnées de la boîte englobante (x1, y1, x2, y2)
        id (int): Identifiant unique de la personne
        confidence (float): Score de confiance de la détection
    
    Returns:
        dict: Dictionnaire contenant les informations de suivi de la personne
    """
    return {
        'bbox': bbox,
        'id': id,
        'confidence': confidence,
        'value': None,
        'disappeared': 0,
        'trajectory': [],
        'crossed_line': False
    }

def get_center(bbox):
    """
    Calcule le point milieu du bas de la bbox.
    
    Args:
        bbox (tuple): Coordonnées de la boîte englobante (x1, y1, x2, y2)
    
    Returns:
        tuple: Coordonnées (x, y) du point central bas
    """
    x1, y1, x2, y2 = map(int, bbox)
    bottom_center_x = x1 + (x2 - x1) // 2
    bottom_center_y = y2
    return (bottom_center_x, bottom_center_y)

def update_person_position(person, bbox):
    """
    Met à jour la position d'une personne et maintient sa trajectoire.
    
    Args:
        person (dict): Dictionnaire de la personne à mettre à jour
        bbox (tuple): Nouvelles coordonnées de la boîte englobante
    
    Notes:
        Conserve uniquement les 30 dernières positions
    """
    person['bbox'] = bbox
    center = get_center(bbox)
    person['trajectory'].append(center)
    if len(person['trajectory']) > 30:
        person['trajectory'].pop(0)

def check_line_crossing(person, line_start, line_end):
    """
    Vérifie si la personne traverse la ligne définie.
    
    Args:
        person (dict): Dictionnaire de la personne à vérifier
        line_start (tuple): Point de départ de la ligne (x, y)
        line_end (tuple): Point d'arrivée de la ligne (x, y)
    
    Returns:
        bool: True si la personne traverse la ligne, False sinon
    """
    if len(person['trajectory']) < 2 or person['crossed_line']:
        return False

    p1 = person['trajectory'][-2]
    p2 = person['trajectory'][-1]

    def ccw(A, B, C):
        """
        Vérifie si les points A, B, C sont dans le sens antihoraire.
        
        Args:
            A, B, C (tuple): Points à vérifier (x, y)
        Returns:
            bool: True si les points sont en sens antihoraire
        """
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    intersect = ccw(p1, line_start, line_end) != ccw(p2, line_start, line_end) and \
                ccw(p1, p2, line_start) != ccw(p1, p2, line_end)

    if intersect:
        person['crossed_line'] = True
        return True
    return False

def create_tracker():
    """
    Crée un dictionnaire contenant l'état initial du tracker.
    
    Returns:
        dict: État initial du tracker avec modèle YOLO chargé
    """
    return {
        'next_id': 1,
        'persons': {},
        'counter': defaultdict(int),
        'model': YOLO(modele_path),
        'crossed_ids': set(),
        'id_mapping': {}
    }

def update_tracker(tracker_state, frame):
    """
    Met à jour l'état du tracker avec une nouvelle frame.
    
    Args:
        tracker_state (dict): État actuel du tracker
        frame (np.array): Image à analyser
    
    Returns:
        list: Liste des personnes actuellement suivies
    
    Notes:
        Utilise ByteTrack pour le suivi et gère la disparition des personnes
        après MAX_DISAPPEAR_FRAMES frames
    """
    results = tracker_state['model'].track(
        source=frame,
        persist=True,
        tracker=bytetrack_path,
        classes=0,
        conf=MIN_CONFIDENCE,
        iou=IOU_THRESHOLD,
        verbose=False
    )

    if results and len(results) > 0 and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        bytetrack_ids = results[0].boxes.id.cpu().numpy()
        
        current_ids = set()
        
        for box, bytetrack_id in zip(boxes, bytetrack_ids):
            bytetrack_id = int(bytetrack_id)
            
            if bytetrack_id not in tracker_state['id_mapping']:
                tracker_state['id_mapping'][bytetrack_id] = tracker_state['next_id']
                tracker_state['next_id'] += 1
            
            sequential_id = tracker_state['id_mapping'][bytetrack_id]
            
            if sequential_id in tracker_state['crossed_ids']:
                continue
                
            current_ids.add(sequential_id)
            
            if sequential_id not in tracker_state['persons']:
                tracker_state['persons'][sequential_id] = create_tracked_person(box, sequential_id, 1.0)
            else:
                update_person_position(tracker_state['persons'][sequential_id], box)
                tracker_state['persons'][sequential_id]['disappeared'] = 0

        for person_id in list(tracker_state['persons'].keys()):
            if person_id not in current_ids:
                tracker_state['persons'][person_id]['disappeared'] += 1
                if tracker_state['persons'][person_id]['disappeared'] > MAX_DISAPPEAR_FRAMES:
                    del tracker_state['persons'][person_id]

    return list(tracker_state['persons'].values())

def mark_as_crossed(tracker_state, person_id):
    """
    Marque une personne comme ayant traversé la ligne et la retire du suivi.
    
    Args:
        tracker_state (dict): État du tracker
        person_id (int): Identifiant de la personne
    """
    tracker_state['crossed_ids'].add(person_id)
    if person_id in tracker_state['persons']:
        del tracker_state['persons'][person_id] 