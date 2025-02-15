import cv2
import numpy as np

def order_points(pts):
    """
    Trie 4 points (x,y) pour qu'ils soient dans l'ordre :
    - haut-gauche, haut-droit, bas-droit, bas-gauche
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # plus petit x+y => haut-gauche
    rect[2] = pts[np.argmax(s)]  # plus grand x+y  => bas-droit

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # plus petit x-y => haut-droit
    rect[3] = pts[np.argmax(diff)]  # plus grand x-y => bas-gauche

    return rect

def detect_macbeth_in_scene(image_path):
    # 1) Lecture de l'image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Impossible de lire l'image : {image_path}")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_black = (0, 0, 0)
    upper_black = (180, 255, 60)
    mask = cv2.inRange(hsv, lower_black, upper_black)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

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

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray_warped, 50, 200)
    found_squares = []

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if w > 10 and h > 10:
                x += 5
                y += 20
                w -= 10
                h -= 40
                found_squares.append((x, y, w, h))

    return found_squares

if __name__ == "__main__":
    image_path = r"D:\Windsuft programme\compteur-de-l-heure\assets\photos\Macbeth\IMG_3673.JPG"
    squares = detect_macbeth_in_scene(image_path)
    print("Coordonnées des carrés détectés :", squares)
