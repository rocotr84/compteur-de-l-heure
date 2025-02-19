import numpy as np
from collections import defaultdict
from config import (
    MAX_DISAPPEAR_FRAMES,
    MIN_CONFIDENCE,
    IOU_THRESHOLD,
    MODEL_PATH,
    BYTETRACK_PATH
)
from ultralytics import YOLO

def create_tracked_person(person_bbox_coords, person_id, person_confidence):
    """
    Crée un dictionnaire représentant une personne suivie.
    
    Args:
        person_bbox_coords (np.ndarray): Coordonnées de la boîte englobante [x1, y1, x2, y2]
        person_id (int): Identifiant unique de la personne
        person_confidence (float): Score de confiance de la détection [0-1]
    
    Returns:
        dict: Dictionnaire contenant:
            - bbox (np.ndarray): Coordonnées de la boîte englobante
            - id (int): Identifiant unique
            - confidence (float): Score de confiance
            - value (None): Réservé pour usage futur
            - frames_disappeared (int): Nombre de frames depuis la dernière détection
            - movement_trajectory (list): Liste des positions [(x,y), ...]
            - has_crossed_line (bool): Indique si la personne a franchi la ligne
    """
    return {
        'bbox': person_bbox_coords,
        'id': person_id,
        'confidence': person_confidence,
        'value': None,
        'frames_disappeared': 0,
        'movement_trajectory': [],
        'has_crossed_line': False
    }

def get_bbox_bottom_center(person_bbox_coords):
    """
    Calcule le point milieu du bas de la bbox.
    
    Args:
        person_bbox_coords (np.ndarray): Coordonnées [x1, y1, x2, y2]
    
    Returns:
        tuple: Point central bas (x, y)
    """
    bbox_coords = np.array(person_bbox_coords)
    center_x = bbox_coords[0] + (bbox_coords[2] - bbox_coords[0]) // 2
    center_y = bbox_coords[3]  # y2 est déjà le point bas
    return (int(center_x), int(center_y))

def update_person_position(person_data, person_bbox_coords):
    """
    Met à jour la position d'une personne et maintient sa trajectoire.
    
    Args:
        person_data (dict): Données de la personne (voir create_tracked_person)
        person_bbox_coords (np.ndarray): Nouvelles coordonnées [x1, y1, x2, y2]
           
    Notes:
        - Met à jour la bbox et la trajectoire
        - Conserve les 30 dernières positions
        - Les positions sont stockées comme (x,y) du point bas central
    """
    person_data['bbox'] = person_bbox_coords
    bbox_center = get_bbox_bottom_center(person_bbox_coords)
    person_data['movement_trajectory'].append(bbox_center)
    if len(person_data['movement_trajectory']) > 30:
        person_data['movement_trajectory'].pop(0)

def check_line_crossing(person_data, counting_line_start, counting_line_end):
    """
    Vérifie si la personne traverse la ligne définie.
    
    Args:
        person_data (dict): Données de la personne (voir create_tracked_person)
        counting_line_start (tuple): Point de départ (x, y)
        counting_line_end (tuple): Point d'arrivée (x, y)
            
    Returns:
        bool: True si la personne traverse la ligne dans cette frame
    
    Notes:
        - Utilise les 2 dernières positions pour détecter l'intersection
        - Une personne ne peut traverser qu'une seule fois (has_crossed_line)
    """
    if len(person_data['movement_trajectory']) < 2 or person_data['has_crossed_line']:
        return False

    # Convertir les points en arrays NumPy pour la vectorisation
    previous_position = np.array(person_data['movement_trajectory'][-2])
    current_position = np.array(person_data['movement_trajectory'][-1])
    line_start = np.array(counting_line_start)
    line_end = np.array(counting_line_end)

    # Calcul vectorisé de l'intersection
    line_vector = line_end - line_start
    movement_vector = current_position - previous_position
    
    # Calcul des vecteurs entre les points
    v_start = previous_position - line_start
    v_end = current_position - line_start

    # Calcul des produits vectoriels
    cross1 = np.cross(line_vector, v_start)
    cross2 = np.cross(line_vector, v_end)
    cross3 = np.cross(movement_vector, line_start - previous_position)
    cross4 = np.cross(movement_vector, line_end - previous_position)

    # Vérification de l'intersection
    if (cross1 * cross2 < 0) and (cross3 * cross4 < 0):
        person_data['has_crossed_line'] = True
        return True
    return False

def create_tracker():
    """
    Crée un dictionnaire contenant l'état initial du tracker.
    
    Returns:
        dict: État initial contenant:
            - next_person_id (int): Prochain ID disponible
            - active_tracked_persons (dict): Personnes actuellement suivies {id: person_data}
            - line_crossing_counter (defaultdict): Compteur de passages {direction: count}
            - person_detection_model (YOLO): Modèle de détection chargé
            - persons_crossed_line (set): IDs des personnes ayant déjà traversé
            - bytetrack_to_internal_ids (dict): Mapping entre IDs ByteTrack et internes
    """
    return {
        'next_person_id': 1,
        'active_tracked_persons': {},
        'line_crossing_counter': defaultdict(int),
        'person_detection_model': YOLO(MODEL_PATH),
        'persons_crossed_line': set(),
        'bytetrack_to_internal_ids': {}
    }

def update_tracker(tracker_state, frame_raw):
    """
    Met à jour l'état du tracker avec une nouvelle frame.
    
    Args:
        tracker_state (dict): État actuel (voir create_tracker)
        frame_raw (np.ndarray): Image BGR à analyser
    
    Returns:
        list: Liste des personnes actuellement suivies
    
    Notes:
        - Utilise ByteTrack pour le suivi
        - Supprime les personnes après MAX_DISAPPEAR_FRAMES frames sans détection
        - Maintient la correspondance entre IDs ByteTrack et IDs internes
        - Ignore les personnes ayant déjà traversé la ligne
    """
    detection_results = tracker_state['person_detection_model'].track(
        source=frame_raw,
        persist=True,
        tracker=BYTETRACK_PATH,
        classes=0,
        conf=MIN_CONFIDENCE,
        iou=IOU_THRESHOLD,
        verbose=False
    )

    if detection_results and len(detection_results) > 0 and detection_results[0].boxes.id is not None:
        detected_bboxes = detection_results[0].boxes.xyxy.cpu().numpy()
        detected_bytetrack_ids = detection_results[0].boxes.id.cpu().numpy()
        
        active_person_ids = set()
        
        for person_bbox, bytetrack_person_id in zip(detected_bboxes, detected_bytetrack_ids):
            bytetrack_person_id = int(bytetrack_person_id)
            
            if bytetrack_person_id not in tracker_state['bytetrack_to_internal_ids']:
                tracker_state['bytetrack_to_internal_ids'][bytetrack_person_id] = tracker_state['next_person_id']
                tracker_state['next_person_id'] += 1
            
            internal_person_id = tracker_state['bytetrack_to_internal_ids'][bytetrack_person_id]
            
            if internal_person_id in tracker_state['persons_crossed_line']:
                continue
                
            active_person_ids.add(internal_person_id)
            
            if internal_person_id not in tracker_state['active_tracked_persons']:
                tracker_state['active_tracked_persons'][internal_person_id] = create_tracked_person(person_bbox, internal_person_id, 1.0)
            else:
                update_person_position(tracker_state['active_tracked_persons'][internal_person_id], person_bbox)
                tracker_state['active_tracked_persons'][internal_person_id]['frames_disappeared'] = 0

        for tracked_person_id in list(tracker_state['active_tracked_persons'].keys()):
            if tracked_person_id not in active_person_ids:
                tracker_state['active_tracked_persons'][tracked_person_id]['frames_disappeared'] += 1
                if tracker_state['active_tracked_persons'][tracked_person_id]['frames_disappeared'] > MAX_DISAPPEAR_FRAMES:
                    del tracker_state['active_tracked_persons'][tracked_person_id]

    return list(tracker_state['active_tracked_persons'].values())

def mark_person_as_crossed(tracker_state, person_id):
    """
    Marque une personne comme ayant traversé la ligne et la retire du suivi.
    
    Args:
        tracker_state (dict): État du tracker (voir create_tracker)
        person_id (int): ID interne de la personne
    
    Notes:
        - Ajoute l'ID à persons_crossed_line
        - Supprime la personne des active_tracked_persons
    """
    tracker_state['persons_crossed_line'].add(person_id)
    if person_id in tracker_state['active_tracked_persons']:
        del tracker_state['active_tracked_persons'][person_id] 