from macbeth_nonlinear_color_correction import corriger_image


image_path = r"D:\Windsuft programme\compteur-de-l-heure\assets\photos\Macbeth\IMG_3673.JPG"
cache_file = r"D:\Windsuft programme\compteur-de-l-heure\Camera_macbeth_main\macbeth_cache.json"
detect_squares = True
image_corrige_path = r"D:\Windsuft programme\compteur-de-l-heure\Camera_macbeth_main\image_co.jpg"  
image_entree = r"D:\Windsuft programme\compteur-de-l-heure\assets\photos\Macbeth\IMG_3673.JPG"

corriger_image(image_path, image_corrige_path, cache_file, detect_squares)
