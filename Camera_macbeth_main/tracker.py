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

def create_tracked_person(person_bbox_coords, person_id, person_confidence):
    """
    Crée un dictionnaire représentant une personne suivie.
    
    Args:
        person_bbox_coords (tuple): Coordonnées de la boîte englobante (x1, y1, x2, y2)
        person_id (int): Identifiant unique de la personne
        person_confidence (float): Score de confiance de la détection
    
    Returns:
        dict: Dictionnaire contenant les informations de suivi de la personne
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
        person_bbox_coords (tuple): Coordonnées de la boîte englobante (x1, y1, x2, y2)
    
    Returns:
        tuple: Coordonnées (x, y) du point central bas
    """
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = map(int, person_bbox_coords)
    center_x = bbox_x1 + (bbox_x2 - bbox_x1) // 2
    center_y = bbox_y2
    return (center_x, center_y)

def update_person_position(person_data, person_bbox_coords):
    """
    Met à jour la position d'une personne et maintient sa trajectoire.
    
    Args:
        person_data (dict): Dictionnaire de la personne à mettre à jour
        person_bbox_coords (tuple): Nouvelles coordonnées de la boîte englobante
           
    Notes:
        Conserve uniquement les 30 dernières positions
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
        person_data (dict): Dictionnaire de la personne à vérifier
        counting_line_start (tuple): Point de départ de la ligne (x, y)
        counting_line_end (tuple): Point d'arrivée de la ligne (x, y)
            
    Returns:
        bool: True si la personne traverse la ligne, False sinon
    """
    if len(person_data['movement_trajectory']) < 2 or person_data['has_crossed_line']:
        return False

    trajectory_point_prev = person_data['movement_trajectory'][-2]
    trajectory_point_curr = person_data['movement_trajectory'][-1]

    def is_counterclockwise(point_1, point_2, point_3):
        """
        Vérifie si trois points forment un virage dans le sens antihoraire.
        
        Args:
            point_1, point_2, point_3 (tuple): Points à vérifier (x, y)
        Returns:
            bool: True si les points sont en sens antihoraire
        """
        return (point_3[1] - point_1[1]) * (point_2[0] - point_1[0]) > \
               (point_2[1] - point_1[1]) * (point_3[0] - point_1[0])

    line_crossing_detected = is_counterclockwise(trajectory_point_prev, counting_line_start, counting_line_end) != \
                           is_counterclockwise(trajectory_point_curr, counting_line_start, counting_line_end) and \
                           is_counterclockwise(trajectory_point_prev, trajectory_point_curr, counting_line_start) != \
                           is_counterclockwise(trajectory_point_prev, trajectory_point_curr, counting_line_end)

    if line_crossing_detected:
        person_data['has_crossed_line'] = True
        return True
    return False

def create_tracker():
    """
    Crée un dictionnaire contenant l'état initial du tracker.
    
    Returns:
        dict: État initial du tracker avec modèle YOLO chargé
    """
    return {
        'next_person_id': 1,
        'active_tracked_persons': {},
        'line_crossing_counter': defaultdict(int),
        'person_detection_model': YOLO(modele_path),
        'persons_crossed_line': set(),
        'bytetrack_to_internal_ids': {}
    }

def update_tracker(tracker_state, frame_raw):
    """
    Met à jour l'état du tracker avec une nouvelle frame.
    
    Args:
        tracker_state (dict): État actuel du tracker
        frame_raw (np.array): Image à analyser
    
    Returns:
        list: Liste des personnes actuellement suivies
    
    Notes:
        Utilise ByteTrack pour le suivi et gère la disparition des personnes
        après MAX_DISAPPEAR_FRAMES frames
    """
    detection_results = tracker_state['person_detection_model'].track(
        source=frame_raw,
        persist=True,
        tracker=bytetrack_path,
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
        tracker_state (dict): État du tracker
        person_id (int): Identifiant de la personne
    """
    tracker_state['persons_crossed_line'].add(person_id)
    if person_id in tracker_state['active_tracked_persons']:
        del tracker_state['active_tracked_persons'][person_id] 