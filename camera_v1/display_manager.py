import cv2
import time
from config import (frame_delay, SHOW_TRAJECTORIES, SAVE_VIDEO, VIDEO_OUTPUT_PATH, 
                   VIDEO_FPS, VIDEO_CODEC, output_width, output_height, DETECTION_MODE)
from color_detector import ColorDetector
from number_detector import NumberDetector

class DisplayManager:
    """
    Classe gérant l'affichage des éléments visuels de l'application.
    Responsable de l'affichage des personnes détectées, des trajectoires,
    des compteurs et des lignes de comptage.
    """
    def __init__(self):
        """
        Initialise le gestionnaire d'affichage
        Définit le nom de la fenêtre d'affichage
        """
        self.window_name = "Tracking"
        self.video_writer = None
        if DETECTION_MODE == "color":
            self.detector = ColorDetector()
        else:
            self.detector = NumberDetector()
        self.start_time = time.time()  # Ajout du temps de départ
        if SAVE_VIDEO:
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
            self.video_writer = cv2.VideoWriter(
                VIDEO_OUTPUT_PATH,
                fourcc,
                VIDEO_FPS,
                (output_width, output_height)
            )
        
    def draw_person(self, frame, person):
        """
        Dessine les éléments visuels pour une personne détectée
        Args:
            frame (np.array): Image sur laquelle dessiner
            person (TrackedPerson): Objet personne contenant les informations à afficher
        """
        # Conversion des coordonnées en entiers
        x1, y1, x2, y2 = map(int, person.bbox)
        
        # Rectangle principal autour de la personne
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Définition et dessin de la ROI (Region of Interest) du t-shirt
        # La ROI est définie comme une zone centrale du haut du corps
        roi_x1 = int(x1 + (x2 - x1) * 0.30)  # 30% depuis la gauche
        roi_x2 = int(x1 + (x2 - x1) * 0.70)  # 70% depuis la gauche
        roi_y1 = int(y1 + (y2 - y1) * 0.2)   # 20% depuis le haut
        roi_y2 = int(y1 + (y2 - y1) * 0.4)   # 40% depuis le haut
        
        # Vérification que la ROI est dans les limites de l'image
        if roi_x1 >= 0 and roi_y1 >= 0 and roi_x2 <= frame.shape[1] and roi_y2 <= frame.shape[0]:
            roi_coords = (roi_x1, roi_y1, roi_x2, roi_y2)
            
            if DETECTION_MODE == "color":
                detected_value = self.detector.get_dominant_color(frame, roi_coords)
                person.value = detected_value
                self.detector.visualize_color(frame, roi_coords, detected_value)
            else:
                detected_value = self.detector.get_number(frame, roi_coords)
                person.value = detected_value
                self.detector.visualize_number(frame, roi_coords, detected_value)
        
        # Affichage de l'ID (en entier) et de la valeur détectée au-dessus de la personne
        id_str = f"{int(person.id)}"  # Conversion en entier
        
        if DETECTION_MODE == "color" and person.value:
            # Traduction des couleurs en français
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
            color_fr = color_translation.get(person.value, person.value)
            label = f"{id_str} - {color_fr}"
        elif DETECTION_MODE == "number" and person.value:
            label = f"{id_str} - N°{person.value}"
        else:
            label = id_str
        
        # Affichage du texte avec un fond noir pour meilleure lisibilité
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        
        # Calcul de la taille du texte pour le fond
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Dessin du fond noir
        cv2.rectangle(frame, 
                     (x1, y1 - text_height - 5), 
                     (x1 + text_width, y1), 
                     (0, 0, 0), 
                     -1)
        
        # Dessin du texte
        cv2.putText(frame, label, 
                    (x1, y1-5), 
                    font, 
                    font_scale, 
                    (255, 255, 255),  # Texte en blanc
                    thickness)
        
        # Dessin de la trajectoire seulement si l'option est activée
        if SHOW_TRAJECTORIES and len(person.trajectory) > 1:
            for i in range(len(person.trajectory)-1):
                cv2.line(frame, 
                        person.trajectory[i],
                        person.trajectory[i+1],
                        (0, 0, 255), 2)
        
        # Dessine le point de référence (point du bas) utilisé pour le comptage
        bottom_center = person.get_center()
        center_point = (int(bottom_center[0]), int(bottom_center[1]))
        
        # Dessine un point plus gros et plus visible
        cv2.circle(frame, 
                   center_point,  
                   1,            # Rayon plus grand
                   (0, 0, 255),  # Couleur rouge vif
                   -1)           # Remplissage
        
    def draw_counters(self, frame, counter):
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
            
    def draw_crossing_line(self, frame, start_point, end_point):
        """
        Dessine la ligne de comptage
        Args:
            frame (np.array): Image sur laquelle dessiner la ligne
            start_point (tuple): Point de début de la ligne (x, y)
            end_point (tuple): Point de fin de la ligne (x, y)
        """
        cv2.line(frame, start_point, end_point, (0, 0, 255), 2)
        
    def draw_timer(self, frame):
        """Affiche le chronomètre sur l'image"""
        elapsed_time = time.time() - self.start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        timer_text = f"{minutes:02d}:{seconds:02d}"
        
        # Position en haut à gauche
        cv2.putText(frame, timer_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return elapsed_time  # Retourne le temps écoulé pour l'utiliser ailleurs

    def show_frame(self, frame):
        """
        Affiche ou enregistre la frame selon la configuration
        Returns:
            bool: True si l'utilisateur a demandé de quitter (touche 'q')
            float: Temps écoulé depuis le début
        """
        elapsed_time = self.draw_timer(frame)
        
        if SAVE_VIDEO:
            self.video_writer.write(frame)
            return False, elapsed_time
        else:
            cv2.imshow("Tracking", frame)
            return cv2.waitKey(1) & 0xFF == ord('q'), elapsed_time

    def release(self):
        """
        Libère les ressources
        """
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows() 