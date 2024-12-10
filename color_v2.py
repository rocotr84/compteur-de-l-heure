import json
from math import sqrt

# Fonction pour charger les couleurs depuis un fichier JSON
def load_colors_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [color['rgb'] for color in data]  # Retourner les couleurs RGB

# Fonction pour trouver la couleur la plus proche
def closest_color(rgb, COLORS):
    if not isinstance(rgb, tuple) or len(rgb) != 3:
        raise ValueError("L'entrée RGB doit être un tuple de 3 entiers.")
    
    r, g, b = rgb
    color_diffs = []
    for color in COLORS:
        cr, cg, cb = color
        # Calcul de la différence de couleur en utilisant la distance euclidienne
        color_diff = sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
        color_diffs.append((color_diff, color))

    # Trouver la couleur la plus proche
    closest = min(color_diffs, key=lambda x: x[0])
    return closest[1]

# Charger les couleurs depuis le fichier JSON
colors = load_colors_from_json('colors.json')

# Définir la couleur à tester
input_rgb = (0, 250, 180)

# Appeler la fonction et afficher la couleur la plus proche
closest = closest_color(input_rgb, colors)
print(f"La couleur la plus proche de {input_rgb} est {closest}")
