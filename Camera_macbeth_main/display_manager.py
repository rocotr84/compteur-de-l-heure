import cv2
import numpy as np
import time
from typing import Optional
from config import (SHOW_ROI_AND_COLOR, SHOW_TRAJECTORIES, 
                   SHOW_CENTER, SHOW_LABELS, SAVE_VIDEO, VIDEO_OUTPUT_PATH, 
                   VIDEO_FPS, VIDEO_CODEC, output_width, output_height, 
                   DETECTION_MODE)
from color_detector import get_dominant_color, visualize_color
#from number_detector import get_number, visualize_number

# Variables globales pour la gestion de l'affichage
display_window_name = "Tracking"
video_output_writer: Optional[cv2.VideoWriter] = None
display_start_time = time.time()

def init_display() -> None:
    """
    Initialise le gestionnaire d'affichage.
    Configure le VideoWriter si l'enregistrement vidéo est activé.
    """
    global video_output_writer
    if SAVE_VIDEO:
        # Utilisation de getattr pour éviter l'erreur de typage
        fourcc_func = getattr(cv2, 'VideoWriter_fourcc')
        video_codec = fourcc_func(*VIDEO_CODEC)
        video_output_writer = cv2.VideoWriter(
            VIDEO_OUTPUT_PATH,
            video_codec,
            VIDEO_FPS,
            (output_width, output_height)
        )

def draw_person(frame_display, tracked_person_data):
    """
    Dessine les éléments visuels pour une personne détectée.
    
    Cette fonction gère l'affichage de :
    1. Rectangle de détection
    2. Zone d'intérêt (ROI) pour la détection de couleur/numéro
    3. Étiquette d'identification
    4. Trajectoire (si activée)
    5. Point central
    
    Args:
        frame_display (np.array): Image sur laquelle dessiner
        tracked_person_data (dict): Informations de la personne
    """
    person_bbox_x1, person_bbox_y1, person_bbox_x2, person_bbox_y2 = map(int, tracked_person_data['bbox'])
    cv2.rectangle(frame_display, 
                 (person_bbox_x1, person_bbox_y1), 
                 (person_bbox_x2, person_bbox_y2), 
                 (0, 255, 0), 2)
    
    # ROI et couleur conditionnels
    if SHOW_ROI_AND_COLOR:
        detection_zone_x1 = int(person_bbox_x1 + (person_bbox_x2 - person_bbox_x1) * 0.30)
        detection_zone_x2 = int(person_bbox_x1 + (person_bbox_x2 - person_bbox_x1) * 0.70)
        detection_zone_y1 = int(person_bbox_y1 + (person_bbox_y2 - person_bbox_y1) * 0.2)
        detection_zone_y2 = int(person_bbox_y1 + (person_bbox_y2 - person_bbox_y1) * 0.4)
        
        if (detection_zone_x1 >= 0 and detection_zone_y1 >= 0 and 
            detection_zone_x2 <= frame_display.shape[1] and 
            detection_zone_y2 <= frame_display.shape[0]):
            
            detection_zone_coords = (detection_zone_x1, detection_zone_y1, 
                                   detection_zone_x2, detection_zone_y2)
            
            detected_value = get_dominant_color(frame_display, detection_zone_coords)
            tracked_person_data['value'] = detected_value
            visualize_color(frame_display, detection_zone_coords, detected_value)
    
    # Éléments visuels optionnels
    if SHOW_LABELS:
        _draw_person_label(frame_display, tracked_person_data, person_bbox_x1, person_bbox_y1)
    if SHOW_TRAJECTORIES:
        _draw_person_trajectory(frame_display, tracked_person_data)
    if SHOW_CENTER:
        _draw_person_center(frame_display, tracked_person_data)

def _draw_person_label(frame_display, tracked_person_data, label_pos_x, label_pos_y):
    """
    Dessine l'étiquette d'identification de la personne.
    
    Affiche l'ID et la valeur détectée (couleur ou numéro) avec un fond noir
    pour une meilleure lisibilité.
    
    Args:
        frame (np.array): Image sur laquelle dessiner
        person (dict): Informations de la personne
        x1 (int): Coordonnée X du coin supérieur gauche
        y1 (int): Coordonnée Y du coin supérieur gauche
    """
    person_id = f"{int(tracked_person_data['id'])}"
    
    if DETECTION_MODE == "color" and tracked_person_data['value']:
        color_display_names = {
            "rouge_fonce": "rouge fonce",
            "bleu_fonce": "bleu fonce",
            "bleu_clair": "bleu clair",
            "vert_fonce": "vert fonce",
            "vert_clair": "vert clair",
            "rose": "rose",
            "jaune": "jaune",
            "blanc": "blanc",
            "noir": "noir",
            "inconnu": "inconnu"
        }
        display_color_name = color_display_names.get(tracked_person_data['value'], tracked_person_data['value'])
        label_text = f"{person_id} - {display_color_name}"
    elif DETECTION_MODE == "number" and tracked_person_data['value']:
        label_text = f"{person_id} - N°{tracked_person_data['value']}"
    else:
        label_text = person_id
        
    font_style = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 2
    
    (text_width, text_height), _ = cv2.getTextSize(label_text, font_style, font_scale, font_thickness)
    
    # Fond noir pour meilleure lisibilité
    cv2.rectangle(frame_display, 
                 (label_pos_x, label_pos_y - text_height - 5), 
                 (label_pos_x + text_width, label_pos_y), 
                 (0, 0, 0), 
                 -1)
    
    cv2.putText(frame_display, label_text, 
                (label_pos_x, label_pos_y-5), 
                font_style, 
                font_scale, 
                (255, 255, 255),
                font_thickness)

def _draw_person_trajectory(frame_display, tracked_person_data):
    """
    Dessine la trajectoire de déplacement de la personne.
    
    Trace des lignes entre les positions successives si SHOW_TRAJECTORIES est activé.
    
    Args:
        frame (np.array): Image sur laquelle dessiner
        person (dict): Informations de la personne incluant sa trajectoire
    """
    if SHOW_TRAJECTORIES and 'trajectory' in tracked_person_data and len(tracked_person_data['trajectory']) > 1:
        trajectory_points = tracked_person_data['trajectory']
        for i in range(len(trajectory_points)-1):
            cv2.line(frame_display, 
                    trajectory_points[i],
                    trajectory_points[i+1],
                    (0, 0, 255), 2)

def _draw_person_center(frame_display, tracked_person_data):
    """
    Dessine le point central bas de la personne.
    
    Place un point rouge au centre bas du rectangle de détection
    pour représenter la position de la personne.
    
    Args:
        frame (np.array): Image sur laquelle dessiner
        person (dict): Informations de la personne contenant:
            - 'bbox': tuple (x1, y1, x2, y2) des coordonnées du rectangle
    
    Notes:
        Le point est dessiné en rouge (BGR: 0, 0, 255) avec un rayon de 1 pixel
    """
    person_bbox_x1, person_bbox_y1, person_bbox_x2, person_bbox_y2 = map(int, tracked_person_data['bbox'])
    person_center_x = (person_bbox_x1 + person_bbox_x2) // 2
    person_center_y = person_bbox_y2  # Point bas
    cv2.circle(frame_display, 
               (person_center_x, person_center_y),
               1,
               (0, 0, 255),
               -1)

def draw_counters(frame_display, counter_values):
    """
    Affiche les compteurs pour chaque valeur détectée.
    
    Affiche une liste verticale des compteurs avec le format "valeur: nombre"
    en haut à gauche de l'image.
    
    Args:
        frame (np.array): Image sur laquelle afficher les compteurs
        counter (defaultdict): Dictionnaire {valeur: nombre} des compteurs
    """
    text_y_position = 30
    for value_name, count in counter_values.items():
        cv2.putText(frame_display, f"{value_name}: {count}", 
                   (10, text_y_position),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        text_y_position += 30

def draw_crossing_line(frame_display, line_start_point, line_end_point):
    """
    Dessine la ligne de comptage sur l'image.
    
    Args:
        frame (np.array): Image sur laquelle dessiner
        start_point (tuple): Point de début (x, y) de la ligne
        end_point (tuple): Point de fin (x, y) de la ligne
    """
    cv2.line(frame_display, line_start_point, line_end_point, (0, 0, 255), 2)

def draw_timer(frame_display):
    """
    Affiche le chronomètre sur l'image.
    
    Affiche le temps écoulé depuis le début au format MM:SS
    en haut à gauche de l'image.
    
    Args:
        frame (np.array): Image sur laquelle afficher le timer
    
    Returns:
        float: Temps écoulé en secondes depuis le début
    """
    elapsed_time = time.time() - display_start_time
    minutes_elapsed = int(elapsed_time // 60)
    seconds_elapsed = int(elapsed_time % 60)
    timer_text = f"{minutes_elapsed:02d}:{seconds_elapsed:02d}"
    
    cv2.putText(frame_display, timer_text, 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return elapsed_time

def show_frame(frame_display: np.ndarray) -> tuple[bool, float]:
    """
    Affiche ou enregistre la frame selon la configuration.
    
    Si SAVE_VIDEO est activé, enregistre la frame dans le fichier vidéo.
    Sinon, affiche la frame dans une fenêtre et vérifie si l'utilisateur
    souhaite quitter (touche 'q').
    
    Args:
        frame_display (np.ndarray): Image à afficher/enregistrer
    
    Returns:
        tuple[bool, float]: (quit_flag, elapsed_time)
    """
    elapsed_time = draw_timer(frame_display)
    
    if SAVE_VIDEO and video_output_writer is not None:
        video_output_writer.write(frame_display)
        return False, elapsed_time
    else:
        cv2.imshow(display_window_name, frame_display)
        return cv2.waitKey(1) & 0xFF == ord('q'), elapsed_time

def release_display() -> None:
    """
    Libère les ressources utilisées par l'affichage.
    
    Ferme le fichier vidéo si l'enregistrement était activé
    et détruit toutes les fenêtres OpenCV.
    """
    if video_output_writer is not None:
        video_output_writer.release()
    cv2.destroyAllWindows() 