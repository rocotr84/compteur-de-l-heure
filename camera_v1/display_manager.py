import cv2
from config import frame_delay, SHOW_TRAJECTORIES, SAVE_VIDEO, VIDEO_OUTPUT_PATH, VIDEO_FPS, VIDEO_CODEC, output_width, output_height

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
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)
        
        # Affichage des informations (ID et couleur) au-dessus de la personne
        label = f"ID: {person.id}"
        if person.color:
            label += f" {person.color}"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Dessin de la trajectoire seulement si l'option est activée
        if SHOW_TRAJECTORIES and len(person.trajectory) > 1:
            for i in range(len(person.trajectory)-1):
                cv2.line(frame, 
                        person.trajectory[i],
                        person.trajectory[i+1],
                        (0, 0, 255), 2)
            
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
        
    def show_frame(self, frame):
        """
        Affiche ou enregistre la frame selon la configuration
        Returns:
            bool: True si l'utilisateur a demandé de quitter (touche 'q')
        """
        if SAVE_VIDEO:
            self.video_writer.write(frame)
            return False  # Continue jusqu'à la fin de la vidéo
        else:
            cv2.imshow("Tracking", frame)
            return cv2.waitKey(1) & 0xFF == ord('q')

    def release(self):
        """
        Libère les ressources
        """
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows() 