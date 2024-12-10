# python color_tracking_yolo.py --video balls.mp4

from collections import deque
import numpy as np
import cv2
import argparse
from ultralytics import YOLO
import imutils

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

# Définir les plages de couleurs HSV
lower = {
    'red': (166, 84, 141), 
    'green': (66, 122, 129), 
    'blue': (97, 100, 117), 
    'yellow': (23, 59, 119), 
    'orange': (0, 50, 80),
    'white': (0, 0, 255)   # HSV pour blanc
}

upper = {
    'red': (186, 255, 255), 
    'green': (86, 255, 255), 
    'blue': (117, 255, 255), 
    'yellow': (54, 255, 255), 
    'orange': (20, 255, 255),

    'white': (180, 255, 255)   # Plage pour blanc (tout lumineux)
}


# Couleurs pour dessiner les cercles
colors = {
    'red': (0, 0, 255), 
    'green': (0, 255, 0), 
    'blue': (255, 0, 0), 
    'yellow': (0, 255, 217), 
    'orange': (0, 140, 255)
}

# Charger le modèle YOLO
model = YOLO("yolo11n.pt")

# Charger la vidéo ou la caméra
if not args.get("video", False):
    video_path = "video.mp4"
    camera = cv2.VideoCapture(video_path)


while True:
    # Lire une trame de la vidéo
    ret, frame = camera.read()
    if not ret:
        break

    # Redimensionner et préparer la trame
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Détection des personnes avec YOLO
    results = model(frame, stream=True)
    for result in results:
        boxes = result.boxes.xyxy
        class_ids = result.boxes.cls

        for box, cls_id in zip(boxes, class_ids):
            if int(cls_id) == 0:  # Classe "person"
                x1, y1, x2, y2 = map(int, box)
                roi_x1 = int(x1 + (x2 - x1) * 0.30)
                roi_x2 = int(x1 + (x2 - x1) * 0.70)
                roi_y1 = int(y1 + (y2 - y1) * 0.2)
                roi_y2 = int(y1 + (y2 - y1) * 0.4)

                if roi_x1 < 0 or roi_y1 < 0 or roi_x2 > frame.shape[1] or roi_y2 > frame.shape[0]:
                    continue

                # Extraire la région du t-shirt
                roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # Appliquer les masques pour détecter la couleur
                max_pixels = 0
                dominant_color = None

                for color_name, (lower_bound, upper_bound) in zip(lower.keys(), zip(lower.values(), upper.values())):
                    mask = cv2.inRange(hsv_roi, np.array(lower_bound), np.array(upper_bound))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
                    pixel_count = cv2.countNonZero(mask)

                    # Trouver la couleur dominante par le nombre de pixels
                    if pixel_count > max_pixels:
                        max_pixels = pixel_count
                        dominant_color = color_name

                # Annoter la couleur dominante sur la région
                if dominant_color:
                    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), colors[dominant_color], 2)
                    cv2.putText(frame, dominant_color, (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[dominant_color], 2)

    # Afficher la trame
    cv2.imshow("Detection de Couleur", frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
