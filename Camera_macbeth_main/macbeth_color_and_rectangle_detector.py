"""
Module de détection et d'analyse de la charte Macbeth.

Ce module permet de :
1. Détecter la charte Macbeth dans une image
2. Corriger la perspective pour obtenir une vue orthogonale
3. Identifier et extraire les 24 carrés de couleur
4. Calculer les couleurs moyennes de chaque carré

La détection utilise le cadre noir de la charte comme repère principal.
"""

import cv2
import numpy as np
import json
import os

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Ordonne 4 points pour former un rectangle cohérent.
    
    Args:
        pts (np.ndarray): Tableau (4,2) de points (x,y)
    
    Returns:
        np.ndarray: Points ordonnés [haut-gauche, haut-droit, bas-droit, bas-gauche]
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # haut-gauche
    rect[2] = pts[np.argmax(s)]  # bas-droit
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # haut-droit
    rect[3] = pts[np.argmax(diff)]  # bas-gauche
    return rect

def detect_macbeth_in_scene(frame_raw: np.ndarray, cache_file: str) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """
    Détecte et analyse la charte Macbeth dans une image.
    
    Le processus comprend :
    1. Détection du cadre noir par seuillage HSV
    2. Correction de la perspective
    3. Détection des 24 carrés internes
    4. Sauvegarde des résultats dans le cache
    
    Args:
        frame_raw (np.ndarray): Image source en BGR
        cache_file (str): Chemin pour sauvegarder les résultats
    
    Returns:
        tuple[np.ndarray, list[tuple[int, int, int, int]]]: (image_redressée, liste_des_carrés)
        
    Raises:
        ValueError: Si l'image est invalide ou si la charte n'est pas détectée
    """
    if frame_raw is None:
        raise ValueError("Image invalide")

    # Détection du cadre noir
    frame_hsv = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2HSV)
    # Conversion des tuples en np.array pour les seuils
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([180, 100, 30], dtype=np.uint8)
    frame_black_mask = cv2.inRange(frame_hsv, lower_black, upper_black)

    # Nettoyage du masque
    morphology_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    frame_black_mask = cv2.morphologyEx(frame_black_mask, cv2.MORPH_CLOSE, morphology_kernel, iterations=1)
    frame_black_mask = cv2.morphologyEx(frame_black_mask, cv2.MORPH_OPEN, morphology_kernel, iterations=1)

    # Recherche du contour du cadre noir
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
        raise ValueError("Aucun grand rectangle noir détecté. Ajustez les seuils ou rapprochez la charte.")

    corner_points_approx = frame_border_contour.reshape(4, 2).astype("float32")
    corner_points_ordered = order_points(corner_points_approx)
    (corner_top_left, corner_top_right, corner_bottom_right, corner_bottom_left) = corner_points_ordered

    # Calcul des dimensions de l'image redressée
    width_bottom = float(np.linalg.norm(corner_bottom_right - corner_bottom_left))
    width_top = float(np.linalg.norm(corner_top_right - corner_top_left))
    target_width = int(max(float(width_bottom), float(width_top)))
    
    height_right = float(np.linalg.norm(corner_top_right - corner_bottom_right))
    height_left = float(np.linalg.norm(corner_top_left - corner_bottom_left))
    target_height = int(max(float(height_right), float(height_left)))

    # Points de destination pour la transformation perspective
    perspective_points_dest = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]
    ], dtype=np.float32)
    
    perspective_matrix = cv2.getPerspectiveTransform(
        corner_points_ordered.astype(np.float32),
        perspective_points_dest
    )
    frame_warped = cv2.warpPerspective(frame_raw, perspective_matrix, (target_width, target_height))

    # Chemins des fichiers de sortie
    warped_image_path = cache_file.replace('.json', '_warped.png')
    annotated_image_path = cache_file.replace('.json', '_warped_with_squares.png')
    cv2.imwrite(warped_image_path, frame_warped)

    # Détection des carrés de couleur
    frame_warped_gray = cv2.cvtColor(frame_warped, cv2.COLOR_BGR2GRAY)
    frame_warped_edges = cv2.Canny(frame_warped_gray, 50, 200)
    color_squares_contours, _ = cv2.findContours(frame_warped_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Détection initiale des carrés
    detected_squares: list[tuple[int, int, int, int]] = []
    squares: list[tuple[int, int, int, int]] = []
    
    for c in color_squares_contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if w > 10 and h > 10:
                # Ajustement des coordonnées pour éviter les bords
                x += 5
                y += 20
                w -= 10
                h -= 40
                detected_squares.append((x, y, w, h))

    # Si on a déjà les 24 carrés, pas besoin de chercher les manquants
    if len(detected_squares) == 24:
        squares = detected_squares
    else:
        # Trier les carrés par position (d'abord y puis x)
        detected_squares.sort(key=lambda s: (s[1], s[0]))
        
        # Regrouper les carrés par ligne (même coordonnée y à 20 pixels près)
        rows = []
        current_row = []
        last_y = None
        
        for square in detected_squares:
            if last_y is None or abs(square[1] - last_y) < 20:
                current_row.append(square)
            else:
                if current_row:
                    rows.append(sorted(current_row, key=lambda s: s[0]))
                    current_row = [square]
            last_y = square[1]
        if current_row:
            rows.append(sorted(current_row, key=lambda s: s[0]))
        
        # Pour chaque ligne, vérifier s'il manque des carrés (la ligne doit comporter 6 carrés)
        for row in rows:
            if len(row) < 6:
                # Calcul de la largeur et hauteur moyennes des carrés de la ligne
                avg_width = sum(s[2] for s in row) / len(row)
                avg_height = sum(s[3] for s in row) / len(row)
                
                # Calcul de l'espacement en x : la distance entre le premier et le dernier carré
                # est divisée en 5 intervalles (pour 6 positions)
                spacing = (row[-1][0] - row[0][0]) / 5
                
                # Générer la liste des positions x attendues pour une ligne complète de 6 carrés
                expected_x = [row[0][0] + i * spacing for i in range(6)]
                
                # Récupérer les positions x existantes
                existing_x = [s[0] for s in row]
                
                # Pour chaque position attendue, vérifier si un carré existe déjà
                for pos in expected_x:
                    if not any(abs(pos - ex) < avg_width/2 for ex in existing_x):
                        # Ajouter le carré manquant avec les dimensions moyennes
                        missing_square = (int(pos), row[0][1], int(avg_width), int(avg_height))
                        row.append(missing_square)
                
                # Retrier la ligne après l'ajout des carrés manquants
                row.sort(key=lambda s: s[0])
            
            squares.extend(row)

    # Création de l'image annotée
    frame_warped_annotated = frame_warped.copy()
    
    for (square_x, square_y, square_width, square_height) in squares:
        cv2.rectangle(frame_warped_annotated, 
                     (square_x, square_y), 
                     (square_x + square_width, square_y + square_height), 
                     (0, 255, 0), 2)
    
    cv2.imwrite(annotated_image_path, frame_warped_annotated)
    
    # Sauvegarde des informations dans le cache (JSON)
    data = {
        "squares": [list(s) for s in squares],
        "warped_image_path": warped_image_path,
        "warped_with_squares_path": annotated_image_path
    }
    with open(cache_file, "w") as f:
        json.dump(data, f)

    return frame_warped, squares

def get_average_colors(frame_raw: np.ndarray, cache_file: str, detect_squares: bool) -> list[tuple[int, int, int]]:
    """
    Calcule les couleurs moyennes des 24 carrés de la charte.
    
    Args:
        frame_raw (np.ndarray): Image source en BGR
        cache_file (str): Chemin du fichier cache
        detect_squares (bool): Si True, détecte les carrés, sinon utilise le cache
    
    Returns:
        list: Liste de 24 tuples (B,G,R) représentant les couleurs moyennes
    
    Raises:
        ValueError: Si le cache est invalide ou si la détection échoue
    """
    if not detect_squares:
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                data = json.load(f)
            squares = [tuple(item) for item in data["squares"]]
            warped_path = data.get("warped_image_path")
            if warped_path and os.path.exists(warped_path):
                frame_warped = cv2.imread(warped_path)
            else:
                raise ValueError("Image transformée non trouvée dans le cache")
        else:
            raise ValueError("Fichier cache non trouvé")
    else:
        frame_warped, squares = detect_macbeth_in_scene(frame_raw, cache_file=cache_file)

    if frame_warped is None:
        raise ValueError("Impossible d'obtenir l'image transformée")

    # Calcul des couleurs moyennes
    colors_average = []
    for (x, y, w, h) in squares:
        roi = frame_warped[y:y+h, x:x+w]
        mean_bgr = cv2.mean(roi)[:3]
        colors_average.append((int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])))
    
    colors_average.reverse()  # Inversion pour correspondre à l'ordre standard
    return colors_average
