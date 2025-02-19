import cv2
import numpy as np
import json
import os
from config import config

class MacbethDetector:
    """Classe responsable de la détection et analyse de la charte Macbeth"""
    
    def __init__(self):
        self.cache_data = {}
        
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Ordonne 4 points pour former un rectangle cohérent"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # haut-gauche
        rect[2] = pts[np.argmax(s)]  # bas-droit
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # haut-droit
        rect[3] = pts[np.argmax(diff)]  # bas-gauche
        return rect

    def detect_macbeth_in_scene(self, frame_raw: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
        """Détecte et analyse la charte Macbeth dans une image"""
        if frame_raw is None:
            raise ValueError("Image invalide")

        # Détection du cadre noir
        frame_hsv = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 100, 30])
        frame_black_mask = cv2.inRange(frame_hsv, lower_black, upper_black)

        # Nettoyage du masque
        morphology_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        frame_black_mask = cv2.morphologyEx(frame_black_mask, cv2.MORPH_CLOSE, morphology_kernel)
        frame_black_mask = cv2.morphologyEx(frame_black_mask, cv2.MORPH_OPEN, morphology_kernel)

        # Détection du contour
        contours_black, _ = cv2.findContours(frame_black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_border_contour = None
        border_area_max = 0
        
        for contour in contours_black:
            contour_area = cv2.contourArea(contour)
            if contour_area > 1000:
                contour_perimeter = cv2.arcLength(contour, True)
                contour_approx = cv2.approxPolyDP(contour, 0.02 * contour_perimeter, True)
                if len(contour_approx) == 4 and contour_area > border_area_max:
                    border_area_max = contour_area
                    frame_border_contour = contour_approx

        if frame_border_contour is None:
            raise ValueError("Aucun grand rectangle noir détecté")

        # Transformation perspective
        corner_points = self._order_points(frame_border_contour.reshape(4, 2))
        
        # Correction du calcul des dimensions cibles
        width1 = np.linalg.norm(corner_points[1] - corner_points[0])
        width2 = np.linalg.norm(corner_points[2] - corner_points[3])
        target_width = int(float(max(width1, width2)))
        
        height1 = np.linalg.norm(corner_points[3] - corner_points[0])
        height2 = np.linalg.norm(corner_points[2] - corner_points[1])
        target_height = int(float(max(height1, height2)))

        dst_points = np.array([[0, 0],
                             [target_width - 1, 0],
                             [target_width - 1, target_height - 1],
                             [0, target_height - 1]], dtype=np.float32)
        
        perspective_matrix = cv2.getPerspectiveTransform(corner_points.astype(np.float32), dst_points)
        frame_warped = cv2.warpPerspective(frame_raw, perspective_matrix, (target_width, target_height))

        # Détection des carrés
        squares = self._detect_color_squares(frame_warped)
        
        return frame_warped, squares

    def _detect_color_squares(self, frame_warped):
        """Détecte les carrés de couleur dans l'image redressée"""
        frame_edges = cv2.Canny(cv2.cvtColor(frame_warped, cv2.COLOR_BGR2GRAY), 50, 200)
        contours, _ = cv2.findContours(frame_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        squares = []
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if w > 10 and h > 10:
                    squares.append((x + 5, y + 20, w - 10, h - 40))
        
        return self._organize_squares(squares)

    def _organize_squares(self, squares):
        """Organise les carrés détectés en grille 4x6"""
        squares.sort(key=lambda s: (s[1], s[0]))
        organized_squares = []
        
        rows = []
        current_row = []
        last_y = None
        
        for square in squares:
            if last_y is None or abs(square[1] - last_y) < 20:
                current_row.append(square)
            else:
                if current_row:
                    rows.append(sorted(current_row, key=lambda s: s[0]))
                current_row = [square]
            last_y = square[1]
            
        if current_row:
            rows.append(sorted(current_row, key=lambda s: s[0]))
            
        for row in rows:
            organized_squares.extend(row)
            
        return organized_squares

    def get_average_colors(self, frame_raw: np.ndarray, detect_squares: bool = True) -> list[tuple[int, int, int]]:
        """Calcule les couleurs moyennes des carrés"""
        frame_warped, squares = self.detect_macbeth_in_scene(frame_raw)
        
        colors = []
        for x, y, w, h in squares:
            roi = frame_warped[y:y+h, x:x+w]
            mean_color = cv2.mean(roi)[:3]
            colors.append(tuple(int(c) for c in mean_color))
            
        colors.reverse()  # Pour correspondre à l'ordre standard
        return colors