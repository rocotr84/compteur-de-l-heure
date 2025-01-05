import cv2
import numpy as np
import easyocr

class NumberDetector:
    """
    Classe pour détecter les numéros sur les t-shirts
    """
    def __init__(self):
        """
        Initialise le détecteur de numéros avec EasyOCR
        """
        # Initialisation du lecteur OCR (uniquement pour les chiffres)
        self.reader = easyocr.Reader(['en'], gpu=True)
        # Configuration pour le prétraitement d'image
        self.min_confidence = 0.6
        
    def preprocess_roi(self, roi):
        """
        Prétraite l'image pour améliorer la détection des numéros
        """
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Amélioration du contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Binarisation adaptative
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Réduction du bruit
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary

    def get_number(self, frame, roi_coords):
        """
        Détecte le numéro dans la ROI
        Args:
            frame (np.array): Image complète
            roi_coords (tuple): Coordonnées de la ROI (x1, y1, x2, y2)
        Returns:
            str: Numéro détecté ou None
        """
        try:
            # Extraction de la ROI
            x1, y1, x2, y2 = roi_coords
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return None

            # Prétraitement de la ROI
            processed_roi = self.preprocess_roi(roi)
            
            # Détection du texte
            results = self.reader.readtext(processed_roi)
            
            # Filtrage des résultats
            for (bbox, text, prob) in results:
                # Vérifier si le texte contient uniquement des chiffres
                if text.isdigit() and prob > self.min_confidence:
                    return text
                    
            return None

        except Exception as e:
            print(f"Erreur lors de la détection du numéro: {str(e)}")
            return None

    def visualize_number(self, frame, roi_coords, number):
        """
        Visualise le numéro détecté sur l'image
        """
        if number is None:
            return
            
        x1, y1, x2, y2 = roi_coords
        
        # Dessine un rectangle autour de la zone de détection
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Affiche le numéro détecté
        cv2.putText(frame, 
                   f"#{number}", 
                   (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.9, 
                   (0, 255, 0), 
                   2)
