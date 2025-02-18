import cv2
import numpy as np
import easyocr
from config import (
    MIN_NUMBER_CONFIDENCE,
    CONTRAST_CLIP_LIMIT,
    CONTRAST_GRID_SIZE,
    BINARY_BLOCK_SIZE,
    BINARY_CONSTANT,
    MORPHOLOGY_KERNEL_SIZE,
    ROI_EXPANSION_RATIO
)

# Paramètres de détection OCR
OCR_READER = easyocr.Reader(['en'], gpu=True)

def preprocess_roi(roi):
    """
    Prétraite l'image pour améliorer la détection des numéros.
    
    Args:
        roi (np.array): Region d'intérêt à prétraiter
        
    Returns:
        np.array: Image prétraitée
    """
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Amélioration du contraste
    clahe = cv2.createCLAHE(clipLimit=CONTRAST_CLIP_LIMIT, 
                           tileGridSize=CONTRAST_GRID_SIZE)
    gray = clahe.apply(gray)
    
    # Binarisation adaptative
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, BINARY_BLOCK_SIZE, BINARY_CONSTANT
    )
    
    # Réduction du bruit
    kernel = np.ones(MORPHOLOGY_KERNEL_SIZE, np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary

def get_number(frame, roi_coords):
    """
    Détecte le numéro dans la ROI.
    
    Args:
        frame (np.array): Image complète
        roi_coords (tuple): Coordonnées de la ROI (x1, y1, x2, y2)
        
    Returns:
        str: Numéro détecté ou None
    """
    try:
        # Extraction de la ROI
        x1, y1, x2, y2 = roi_coords
        
        # Agrandir la ROI selon le ratio défini
        width = x2 - x1
        height = y2 - y1
        
        # Ajustement des coordonnées avec ROI_EXPANSION_RATIO
        x1 = max(0, x1 - int(width * ROI_EXPANSION_RATIO))
        y1 = max(0, y1 - int(height * ROI_EXPANSION_RATIO))
        x2 = min(frame.shape[1], x2 + int(width * ROI_EXPANSION_RATIO))
        y2 = min(frame.shape[0], y2 + int(height * ROI_EXPANSION_RATIO))
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            print("[DEBUG] ROI vide, aucune image à traiter.")
            return None

        # Prétraitement de la ROI
        processed_roi = preprocess_roi(roi)
        print("[DEBUG] ROI prétraitée pour la détection.")
        
        # Détection du texte avec OCR_READER
        results = OCR_READER.readtext(processed_roi)
        print(f"[DEBUG] Résultats de la détection : {results}")
        
        # Filtrage des résultats selon MIN_NUMBER_CONFIDENCE
        for (bbox, text, prob) in results:
            print(f"[DEBUG] Texte détecté : {text}, Probabilité : {prob}")
            if text.isdigit() and prob > MIN_NUMBER_CONFIDENCE:
                print(f"[DEBUG] Numéro détecté : {text}")
                return text
                
        print("[DEBUG] Aucun numéro valide détecté.")
        return None

    except Exception as e:
        print(f"[ERROR] Erreur lors de la détection du numéro: {str(e)}")
        return None

def visualize_number(frame, roi_coords, number):
    """
    Visualise le numéro détecté sur l'image.
    
    Args:
        frame (np.array): Image sur laquelle dessiner
        roi_coords (tuple): Coordonnées de la ROI (x1, y1, x2, y2)
        number (str): Numéro détecté
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
