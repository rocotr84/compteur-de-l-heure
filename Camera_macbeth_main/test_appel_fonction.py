import cv2
from macbeth_nonlinear_color_correction import corriger_image

# Lecture de la vidéo
video_capture = cv2.VideoCapture(r"D:\Windsuft programme\compteur-de-l-heure\assets\video\p3_macbeth.MP4")
cache_file = r"D:\Windsuft programme\compteur-de-l-heure\Camera_macbeth_main\macbeth_cache.json"
detect_squares = True

while video_capture.isOpened():
    ret, current_frame = video_capture.read()
    if not ret:
        break

    # Correction de l'image
    image_corrigee = corriger_image(current_frame, cache_file, detect_squares)

    # Affichage du résultat
    cv2.imshow("Vidéo Corrigée", image_corrigee)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()