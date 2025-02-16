import cv2
from macbeth_nonlinear_color_correction import corriger_image

# Lecture de l'image
image = cv2.imread(r"D:\Windsuft programme\compteur-de-l-heure\assets\photos\Macbeth\IMG_3673.JPG")
cache_file = r"D:\Windsuft programme\compteur-de-l-heure\Camera_macbeth_main\macbeth_cache.json"
detect_squares = True

# Correction de l'image
image_corrigee = corriger_image(image, cache_file, detect_squares)

# Affichage optionnel
cv2.imshow("Image Corrigée", image_corrigee)
cv2.waitKey(0)
cv2.destroyAllWindows()