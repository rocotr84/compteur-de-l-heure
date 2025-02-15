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

def detect_macbeth_in_scene(image_path, detect_squares=True, cache_file="macbeth_cache.json"):
    """
    Détecte la charte Macbeth dans l'image, corrige la perspective,
    et détecte les carrés internes.
    
    Paramètres :
      - image_path : chemin vers l'image source.
      - detect_squares : si True, recherche et stocke les coordonnées des carrés.
                         si False, tente de récupérer ces coordonnées depuis le fichier cache.
      - cache_file : nom du fichier de cache pour stocker les données des carrés.
    
    Renvoie :
      - warped : l'image redressée (la charte).
      - squares : une liste de tuples (x, y, w, h) définissant les zones des carrés.
    """
    # Lecture de l'image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Impossible de lire l'image : {image_path}")

    # Conversion en HSV et création d'un masque pour le fond noir
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_black = (0, 0, 0)
    upper_black = (180, 255, 60)
    mask = cv2.inRange(hsv, lower_black, upper_black)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

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

    squares = []
    if detect_squares:
        # Recherche des carrés dans l'image redressée
        gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray_warped, 50, 200)
        cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if w > 10 and h > 10:
                    # Ajustement pour correspondre à la zone interne du carré
                    x += 5
                    y += 20
                    w -= 10
                    h -= 40
                    squares.append((x, y, w, h))
        # Stocker les coordonnées dans le fichier cache
        data = {"squares": [list(s) for s in squares]}
        with open(cache_file, "w") as f:
            json.dump(data, f)
    else:
        # Récupérer les coordonnées depuis le fichier cache si disponible
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                data = json.load(f)
            squares = [tuple(item) for item in data["squares"]]
        else:
            # Si le cache n'existe pas, on effectue la détection et on crée le cache.
            gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(gray_warped, 50, 200)
            cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    if w > 10 and h > 10:
                        x += 5
                        y += 20
                        w -= 10
                        h -= 40
                        squares.append((x, y, w, h))
            data = {"squares": [list(s) for s in squares]}
            with open(cache_file, "w") as f:
                json.dump(data, f)

    return warped, squares

def get_average_colors(image_path, cache_file, detect_squares):
    """
    Pour chaque carré détecté (ou chargé depuis le cache) dans l'image,
    calcule et retourne la couleur moyenne en format BGR.
    
    Renvoie une liste de tuples (B, G, R) pour chaque carré.
    """
    warped, squares = detect_macbeth_in_scene(image_path, detect_squares=detect_squares, cache_file=cache_file)
    avg_colors = []
    for (x, y, w, h) in squares:
        roi = warped[y:y+h, x:x+w]
        mean_bgr = cv2.mean(roi)[:3]  # Moyenne en BGR
        avg_colors.append((int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])))
    avg_colors.reverse()
    return avg_colors


#if __name__ == "__main__":

    #Pour tester la fonction uniquement
    image_path = r"D:\Windsuft programme\compteur-de-l-heure\assets\photos\Macbeth\IMG_3673.JPG"
    cache_file = r"D:\Windsuft programme\compteur-de-l-heure\Tests_color\macbeth_cache.json"
    detect_squares = False

    warped, squares = detect_macbeth_in_scene(image_path, detect_squares=detect_squares, cache_file=cache_file)
    #print("squares :", squares)

    colors = get_average_colors(image_path, cache_file=cache_file, detect_squares=detect_squares)
    #print("Couleurs moyennes (BGR) :", colors)

