import cv2
import numpy as np
import json
import os

def order_points(pts):
    """
    Trie 4 points (x,y) pour qu'ils soient dans l'ordre :
    - haut-gauche, haut-droit, bas-droit, bas-gauche
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # haut-gauche
    rect[2] = pts[np.argmax(s)]  # bas-droit
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # haut-droit
    rect[3] = pts[np.argmax(diff)]  # bas-gauche
    return rect

def detect_macbeth_in_scene(image, cache_file):
    """
    Détecte la charte Macbeth dans l'image, corrige la perspective,
    et détecte les carrés internes.
    
    Paramètres :
      - image : image source en format BGR (np.array)
      - cache_file : nom du fichier de cache pour stocker les données des carrés
    """
    if image is None:
        raise ValueError("Image invalide")

    # Conversion en HSV et création d'un masque pour le fond noir
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_black = (0, 0, 0)
    upper_black = (180, 100, 30)  # Réduction de la tolérance sur S et V
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # Augmentation de la taille du kernel pour mieux filtrer le bruit
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Recherche du plus grand contour noir (supposé être le cadre de la charte)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_contour = None
    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 1000:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and area > max_area:
                max_area = area
                best_contour = approx

    if best_contour is None:
        raise ValueError("Aucun grand rectangle noir détecté. Ajustez les seuils ou rapprochez la charte.")

    approx_points = best_contour.reshape(4, 2).astype("float32")
    rect = order_points(approx_points)
    (tl, tr, br, bl) = rect

    # Calcul des dimensions de l'image redressée
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    # Transformation perspective
    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Sauvegarder l'image avec les carrés dessinés
    warped_image_path = cache_file.replace('.json', '_warped.png')
    warped_with_squares_path = cache_file.replace('.json', '_warped_with_squares.png')

    # Sauvegarde de l'image redressée
    cv2.imwrite(warped_image_path, warped)


    # Recherche des carrés dans l'image redressée
    squares = []
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray_warped, 50, 200)
    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Détection initiale des carrés
    detected_squares = []
    for c in cnts:
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

    # Créer une copie de l'image warped pour le dessin
    warped_with_squares = warped.copy()
    
    for (x, y, w, h) in squares:
        # Dessiner le rectangle sur l'image
        cv2.rectangle(warped_with_squares, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Sauvegarder l'image avec les carrés dessinés
    cv2.imwrite(warped_with_squares_path, warped_with_squares)
    
    # Sauvegarde des informations dans le cache (JSON)
    data = {
        "squares": [list(s) for s in squares],
        "warped_image_path": warped_image_path,
        "warped_with_squares_path": warped_with_squares_path
    }
    with open(cache_file, "w") as f:
        json.dump(data, f)

    return warped, squares

def get_average_colors(image, cache_file, detect_squares):
    """
    Pour chaque carré détecté dans l'image, calcule et retourne la couleur moyenne en BGR.

    Args:
        image (np.array): Image source au format BGR
        cache_file (str): Chemin vers le fichier de cache
        detect_squares (bool): Si True, détecte les carrés, sinon utilise le cache

    Returns:
        list: Liste de tuples (B, G, R) pour chaque carré
    """
    if not detect_squares:
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                data = json.load(f)
            squares = [tuple(item) for item in data["squares"]]
            # Chargement de l'image redressée depuis le cache
            warped_path = data.get("warped_image_path")
            if warped_path and os.path.exists(warped_path):
                warped = cv2.imread(warped_path)
            else:
                raise ValueError("Image transformée non trouvée dans le cache")
        else:
            raise ValueError("Fichier cache non trouvé")
    else:
        warped, squares = detect_macbeth_in_scene(image, cache_file=cache_file)

    if warped is None:
        raise ValueError("Impossible d'obtenir l'image transformée")

    avg_colors = []
    for (x, y, w, h) in squares:
        roi = warped[y:y+h, x:x+w]
        mean_bgr = cv2.mean(roi)[:3]
        avg_colors.append((int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])))
    # Inverser la liste si besoin (par exemple, si l'ordre de la charte doit être inversé)
    avg_colors.reverse()
    return avg_colors
