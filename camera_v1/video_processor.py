import cv2
import numpy as np

class VideoProcessor:
    """
    Processeur de flux vidéo pour l'analyse des coureurs.
    
    Gère le traitement des images vidéo, incluant le redimensionnement,
    l'application de masques et les ajustements d'image.

    Attributes:
        output_width (int): Largeur cible des images traitées
        output_height (int): Hauteur cible des images traitées
        desired_fps (int): FPS souhaité pour le traitement
        mask (np.ndarray): Masque binaire pour filtrer les zones d'intérêt

    Notes:
        - Le masque permet d'exclure certaines zones de l'image du traitement
        - Les dimensions de sortie sont fixes pour assurer la cohérence du traitement
    """
    def __init__(self, output_width=1280, output_height=720, desired_fps=30):
        """
        Initialise le processeur vidéo avec les paramètres de sortie souhaités
        Args:
            output_width (int): Largeur souhaitée pour les frames de sortie
            output_height (int): Hauteur souhaitée pour les frames de sortie
            desired_fps (int): Nombre d'images par seconde souhaité
        """
        self.output_width = output_width
        self.output_height = output_height
        self.desired_fps = desired_fps
        self.mask = None  # Masque pour filtrer certaines zones de l'image

    def load_mask(self, mask_path):
        """
        Charge et prépare le masque pour le traitement vidéo
        Args:
            mask_path (str): Chemin vers le fichier de masque
        
        Note:
            Le masque est utilisé pour exclure certaines zones de l'image du traitement
            En cas d'échec du chargement, un masque blanc (tout visible) est créé
        """
        try:
            # Chargement du masque en niveaux de gris
            self.mask = cv2.imread(mask_path, 0)
            
            if self.mask is None:
                print(f"Impossible de charger le masque: {mask_path}")
                # Création d'un masque blanc par défaut
                self.mask = np.ones((self.output_height, self.output_width), dtype=np.uint8) * 255
            else:
                # Redimensionnement du masque aux dimensions souhaitées
                self.mask = cv2.resize(self.mask, (self.output_width, self.output_height))
                print(f"Masque chargé avec succès. Taille: {self.mask.shape}")
        except Exception as e:
            print(f"Erreur lors du chargement du masque: {e}")
            # Création d'un masque blanc en cas d'erreur
            self.mask = np.ones((self.output_height, self.output_width), dtype=np.uint8) * 255

    def setup_video_capture(self, video_path):
        """
        Configure la capture vidéo avec les paramètres souhaités
        Args:
            video_path (str): Chemin vers le fichier vidéo
        Returns:
            cv2.VideoCapture: Objet de capture vidéo configuré
        Raises:
            ValueError: Si la vidéo ne peut pas être ouverte
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Erreur : Impossible d'accéder à la vidéo.")
        
        # Configuration du FPS souhaité
        cap.set(cv2.CAP_PROP_FPS, self.desired_fps)
        return cap

    def process_frame(self, frame):
        """
        Traite une frame individuelle
        Args:
            frame (np.array): Image à traiter
        Returns:
            np.array: Image traitée
            
        Note:
            Le traitement inclut :
            - Redimensionnement aux dimensions souhaitées
            - Application du masque si disponible
        """
        # Redimensionnement de la frame
        frame = cv2.resize(frame, (self.output_width, self.output_height))
        
        # Application du masque si disponible
        if self.mask is not None:
            # Vérification de la compatibilité des dimensions
            if self.mask.shape[:2] != frame.shape[:2]:
                self.mask = cv2.resize(self.mask, (frame.shape[1], frame.shape[0]))
            # Application du masque par opération bit à bit
            frame = cv2.bitwise_and(frame, frame, mask=self.mask)
            
        return frame

    def adjust_brightness(self, img, factor=1.2):
        """
        Ajuste la luminosité d'une image.

        Args:
            img (np.ndarray): Image source en format BGR
            factor (float): Facteur de multiplication de la luminosité
                          >1 pour éclaircir, <1 pour assombrir

        Returns:
            np.ndarray: Image avec luminosité ajustée

        Notes:
            L'ajustement est effectué dans l'espace HSV pour préserver
            la teinte et la saturation des couleurs.
        """
        # Conversion en HSV pour modifier la luminosité
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Ajustement de la composante V (luminosité)
        v = np.clip(v * factor, 0, 255).astype(np.uint8)
        
        # Reconstitution de l'image
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) 