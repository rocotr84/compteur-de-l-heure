from ultralytics import YOLO
import cv2
#ligne


# Charger le modèle YOLOv8
model = YOLO("yolov8n.pt")  # Modèle léger pour la détection

# Définir la ligne fixe (exemple : ligne horizontale au centre de l'image)
line_position = 450  # Position verticale de la ligne (pixels)
line_start = (0, line_position)  # Début de la ligne (x=0)
line_end = (800, line_position)  # Fin de la ligne (x=800, largeur de l'image)
#test
# Capture vidéo depuis la webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened(): 
    print("Erreur : Impossible d'accéder à la caméra.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire le flux vidéo.")
        break

    # Détection des objets dans le frame
    results = model(frame, stream=True)
    for result in results:
        boxes = result.boxes.xyxy  # Coordonnées des boîtes (xmin, ymin, xmax, ymax)
        confidences = result.boxes.conf  # Confiances
        class_ids = result.boxes.cls  # IDs des classes

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            # Filtrer uniquement les personnes (classe "person", ID=0 dans COCO)
            if int(cls_id) == 0:  # Classe "person"
                x1, y1, x2, y2 = map(int, box)
                
                # Dessiner la boîte englobante
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Person {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Vérifier si la boîte croise la ligne
                box_center_y = (y1 + y2) // 2  # Centre vertical de la boîte
                if y1 < line_position < y2:  # La ligne est dans les limites de la boîte
                    cv2.putText(frame, "PASSAGE DETECTE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print(f"Passage détecté ! {x1, y1, x2, y2}")

    # Dessiner la ligne fixe sur l'image
    cv2.line(frame, line_start, line_end, (0, 0, 255), 2)

    # Afficher le flux vidéo avec les annotations
    cv2.imshow("Detection avec ligne fixe", frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
