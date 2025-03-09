# Système de Détection et Comptage de Coureurs

## 1. Vue d'ensemble

Le système est conçu pour détecter, suivre et compter les coureurs traversant une ligne virtuelle, tout en identifiant la couleur de leur tenue.

### Technologies principales utilisées

- **OpenCV**: Traitement d'image et visualisation
- **YOLO (You Only Look Once)**: Détection d'objets
- **ByteTrack**: Algorithme de suivi
- **NumPy**: Calculs matriciels
- **Torch**: Support GPU pour l'inférence

## 2. Pipeline de traitement

### 2.1 Acquisition et prétraitement de l'image

- **VideoProcessor**: Gère le flux vidéo
  - Redimensionne les images à une taille standard (1280x720)
  - Applique un masque pour exclure les zones non pertinentes
  - Ajuste la luminosité si nécessaire

### 2.2 Détection et suivi

- **PersonTracker**: Coordonne la détection et le suivi
  - Utilise YOLO pour la détection des personnes
  - Implémente ByteTrack pour le suivi temporel
  - Maintient des IDs uniques pour chaque coureur

### 2.3 Analyse des couleurs

- **ColorDetector**: Analyse les couleurs des tenues
  - Définit une ROI sur le haut du corps
  - Utilise l'espace colorimétrique HSV
  - Applique une pondération temporelle pour stabiliser la détection

### 2.4 Gestion des traversées

- **ColorHistory**: Gère l'historique des passages
  - Enregistre les couleurs détectées par ID
  - Sauvegarde les passages dans un CSV
  - Maintient des statistiques par couleur

## 3. Installation et Configuration

### 3.1 Prérequis

- Installation de Python 3.8 ou supérieur
- python -m pip install --upgrade pip
- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
- pip install torch torchvision torchaudio
- pip install ultralytics # Pour YOLO
- pip install opencv-python-headless
- pip install numpy
- pip install scikit-learn #

### 3.2 Configuration

Le système utilise un fichier `config.py` pour centraliser les paramètres :

- Dimensions de sortie (1280x720)
- Seuils de détection
- Paramètres de tracking
- Configuration des couleurs
- Points de la ligne de comptage

## 4. Fonctionnalités

### 4.1 Détection des couleurs

- Noir
- Blanc
- Rouge foncé
- Bleu foncé/clair
- Vert foncé/clair
- Rose
- Jaune

### 4.2 Suivi et comptage

- Attribution d'ID uniques
- Suivi temporel des trajectoires
- Détection des franchissements de ligne
- Comptage par couleur

### 4.3 Sorties

- Flux vidéo annoté en temps réel
- Fichier CSV avec historique des passages
- Statistiques par couleur
- Chronomètre pour chaque passage

## 5. Améliorations possibles

### 5.1 Performance

- Implémentation d'un traitement multi-thread
- Optimisation du pipeline GPU
- Mise en cache des détections

### 5.2 Robustesse

- Ajout de filtres Kalman pour le suivi
- Amélioration de la détection des couleurs par deep learning
- Gestion des occlusions

### 5.3 Fonctionnalités

- Interface web pour la configuration
- API REST pour l'accès aux données
- Support multi-caméras synchronisé
- Système de replay instantané
- Export des statistiques en temps réel
- Reconnaissance des dossards par OCR

Je recommande YOLOv10n

# Moyennes pour la catégorie "Facile"

yolov8n: 91.71% / 0.0318s
yolov8x: 93.36% / 0.0455s
yolov9t: 91.17% / 0.0599s
yolov9e: 93.99% / 0.0856s
yolov10n: 93.63% / 0.0378s
yolov10x: 95.54% / 0.0567s
yolo11n: 91.98% / 0.0361s
yolo11x: 93.53% / 0.0614s
yolo11s: 93.67% / 0.0351s

# Moyennes pour la catégorie "moyen"

yolov8n: 90.03% / 0.0402s
yolov8x: 91.77% / 0.0520s
yolov9t: 91.88% / 0.0694s
yolov9e: 92.90% / 0.1053s
yolov10n: 89.66% / 0.0463s
yolov10x: 94.92% / 0.0689s
yolo11n: 89.15% / 0.0437s
yolo11x: 91.81% / 0.0625s
yolo11s: 90.76% / 0.0500s

# Moyennes pour la catégorie "dur"

yolov8n: 86.12% (précision) / 0.0287s (temps) / 50% (taux de réussite)
yolov8x: 84.64% / 0.0449s / 30% (taux de réussite)
yolov9t: 89.75% / 0.0571s / 60% (taux de réussite)
yolov9e: 87.13% / 0.0813s / 20% (taux de réussite)
yolov10n: 92.68% / 0.0371s / 70% (taux de réussite)
yolov10x: 85.08% / 0.0619s / 20% (taux de réussite)
yolo11n: 86.54% / 0.0333s / 50% (taux de réussite)
yolo11x: 82.61% / 0.0580s / 10% (taux de réussite)
yolo11s: 86.31% / 0.0347s / 40% (taux de réussite)
