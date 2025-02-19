import cv2
import torch
import numpy as np
from ultralytics import YOLO
from config import config

class ObjectDetector:
    """
    Classe responsable de la détection des objets dans les images
    """
    def __init__(self):
        try:
            self.model = YOLO(config.MODEL_PATH)
        except Exception as e:
            print(f"Erreur lors du chargement du modèle YOLO: {e}")
            self.model = None
        self.detection_mask = cv2.imread(config.DETECTION_MASK_PATH, cv2.IMREAD_GRAYSCALE)
        if self.detection_mask is not None:
            self.detection_mask = cv2.resize(
                self.detection_mask, 
                (config.output_width, config.output_height)
            )

    def detect_objects(self, frame):
        """
        Détecte les objets dans une image
        
        Args:
            frame: Image en format BGR
            
        Returns:
            list: Liste des détections (x1, y1, x2, y2, confidence, class_id)
        """
        if self.detection_mask is not None:
            # Appliquer le masque à l'image
            masked_frame = cv2.bitwise_and(frame, frame, mask=self.detection_mask)
        else:
            masked_frame = frame

        # Effectuer la détection
        results = self.model(
            masked_frame, 
            conf=config.MIN_CONFIDENCE, 
            verbose=False
        )[0]

        # Convertir les résultats en format standard
        detections = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = result
            if confidence >= config.MIN_CONFIDENCE:
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': confidence,
                    'class_id': int(class_id)
                })

        return detections

    def draw_detections(self, frame, detections, colors=None):
        """
        Dessine les boîtes de détection sur l'image
        
        Args:
            frame: Image en format BGR
            detections: Liste des détections
            colors: Dictionnaire des couleurs par classe (optionnel)
        
        Returns:
            np.ndarray: Image avec les détections dessinées
        """
        if colors is None:
            colors = {
                0: (0, 255, 0),  # Vert par défaut
            }

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            confidence = det['confidence']

            color = colors.get(class_id, (0, 255, 0))
            
            # Dessiner la boîte
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Ajouter le label avec la confiance
            label = f"Class {class_id}: {confidence:.2f}"
            cv2.putText(
                frame, 
                label, 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                2
            )

        return frame

    def get_roi(self, frame, bbox, expansion_ratio=None):
        """
        Extrait une région d'intérêt (ROI) de l'image
        
        Args:
            frame: Image en format BGR
            bbox: Tuple (x1, y1, x2, y2) définissant la boîte
            expansion_ratio: Ratio d'expansion de la ROI (optionnel)
            
        Returns:
            tuple: (ROI, coordonnées_originales)
        """
        if expansion_ratio is None:
            expansion_ratio = config.ROI_EXPANSION_RATIO

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        # Calculer l'expansion
        dw = int(w * expansion_ratio)
        dh = int(h * expansion_ratio)
        
        # Nouvelles coordonnées avec expansion
        new_x1 = max(0, x1 - dw)
        new_y1 = max(0, y1 - dh)
        new_x2 = min(frame.shape[1], x2 + dw)
        new_y2 = min(frame.shape[0], y2 + dh)
        
        # Extraire la ROI
        roi = frame[new_y1:new_y2, new_x1:new_x2]
        
        return roi, (new_x1, new_y1, new_x2, new_y2) 