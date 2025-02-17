import cv2
import time
from config import (frame_delay, SHOW_TRAJECTORIES, SAVE_VIDEO, VIDEO_OUTPUT_PATH, 
                   VIDEO_FPS, VIDEO_CODEC, output_width, output_height, DETECTION_MODE)
from color_detector import get_dominant_color, visualize_color
from number_detector import get_number, visualize_number

# Variables globales pour remplacer les attributs de classe
window_name = "Tracking"
video_writer = None
start_time = time.time()

def init_display():
    """Initialise le gestionnaire d'affichage"""
    global video_writer
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        video_writer = cv2.VideoWriter(
            VIDEO_OUTPUT_PATH,
            fourcc,
            VIDEO_FPS,
            (output_width, output_height)
        )

def draw_person(frame, person):
    """
    Dessine les éléments visuels pour une personne détectée
    Args:
        frame (np.array): Image sur laquelle dessiner
        person (dict): Dictionnaire contenant les informations de la personne
    """
    x1, y1, x2, y2 = map(int, person['bbox'])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    roi_x1 = int(x1 + (x2 - x1) * 0.30)
    roi_x2 = int(x1 + (x2 - x1) * 0.70)
    roi_y1 = int(y1 + (y2 - y1) * 0.2)
    roi_y2 = int(y1 + (y2 - y1) * 0.4)
    
    if roi_x1 >= 0 and roi_y1 >= 0 and roi_x2 <= frame.shape[1] and roi_y2 <= frame.shape[0]:
        roi_coords = (roi_x1, roi_y1, roi_x2, roi_y2)
        
        if DETECTION_MODE == "color":
            detected_value = get_dominant_color(frame, roi_coords)
            person['value'] = detected_value
            visualize_color(frame, roi_coords, detected_value)
        else:
            detected_value = get_number(frame, roi_coords)
            person['value'] = detected_value
            visualize_number(frame, roi_coords, detected_value)
    
    _draw_person_label(frame, person, x1, y1)
    _draw_person_trajectory(frame, person)
    _draw_person_center(frame, person)

def _draw_person_label(frame, person, x1, y1):
    """Dessine l'étiquette de la personne"""
    id_str = f"{int(person['id'])}"
    
    if DETECTION_MODE == "color" and person['value']:
        color_translation = {
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
        color_fr = color_translation.get(person['value'], person['value'])
        label = f"{id_str} - {color_fr}"
    elif DETECTION_MODE == "number" and person['value']:
        label = f"{id_str} - N°{person['value']}"
    else:
        label = id_str
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
    
    cv2.rectangle(frame, 
                 (x1, y1 - text_height - 5), 
                 (x1 + text_width, y1), 
                 (0, 0, 0), 
                 -1)
    
    cv2.putText(frame, label, 
                (x1, y1-5), 
                font, 
                font_scale, 
                (255, 255, 255),
                thickness)

def _draw_person_trajectory(frame, person):
    """Dessine la trajectoire de la personne"""
    if SHOW_TRAJECTORIES and 'trajectory' in person and len(person['trajectory']) > 1:
        for i in range(len(person['trajectory'])-1):
            cv2.line(frame, 
                    person['trajectory'][i],
                    person['trajectory'][i+1],
                    (0, 0, 255), 2)

def _draw_person_center(frame, person):
    """Dessine le point central de la personne"""
    x1, y1, x2, y2 = map(int, person['bbox'])
    center_x = (x1 + x2) // 2
    center_y = y2  # Point du bas
    cv2.circle(frame, 
               (center_x, center_y),
               1,
               (0, 0, 255),
               -1)

def draw_counters(frame, counter):
    """
    Affiche les compteurs pour chaque couleur
    Args:
        frame (np.array): Image sur laquelle afficher les compteurs
        counter (defaultdict): Dictionnaire contenant les compteurs par couleur
    """
    y_offset = 30  # Position verticale initiale
    for color, count in counter.items():
        cv2.putText(frame, f"{color}: {count}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        y_offset += 30  # Espacement vertical entre chaque ligne
        
def draw_crossing_line(frame, start_point, end_point):
    """
    Dessine la ligne de comptage
    Args:
        frame (np.array): Image sur laquelle dessiner la ligne
        start_point (tuple): Point de début de la ligne (x, y)
        end_point (tuple): Point de fin de la ligne (x, y)
    """
    cv2.line(frame, start_point, end_point, (0, 0, 255), 2)
    
def draw_timer(frame):
    """Affiche le chronomètre sur l'image"""
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    timer_text = f"{minutes:02d}:{seconds:02d}"
    
    cv2.putText(frame, timer_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return elapsed_time

def show_frame(frame):
    """Affiche ou enregistre la frame selon la configuration"""
    elapsed_time = draw_timer(frame)
    
    if SAVE_VIDEO:
        video_writer.write(frame)
        return False, elapsed_time
    else:
        cv2.imshow("Tracking", frame)
        return cv2.waitKey(1) & 0xFF == ord('q'), elapsed_time

def release():
    """Libère les ressources"""
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows() 