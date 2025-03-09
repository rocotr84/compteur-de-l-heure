# Système de détection et suivi avec correction Macbeth

## Introduction

Ce système permet la détection et le suivi de personnes dans une vidéo, avec correction des couleurs basée sur une charte Macbeth. Il identifie les couleurs dominantes des personnes et enregistre les passages à travers une ligne définie.

## Fonctionnalités principales

- Détection et suivi de personnes en temps réel
- Correction des couleurs via une charte Macbeth
- Identification des couleurs dominantes
- Comptage des passages par couleur
- Enregistrement des données en CSV ou SQLite

## Architecture

Le système est composé de plusieurs modules organisés autour d'une classe `Application` centrale:

- **Application**: Orchestration générale du système
- **Tracker**: Détection et suivi des personnes
- **Video Processor**: Traitement et correction des images
- **Display Manager**: Affichage et visualisation
- **Detection History**: Enregistrement des détections
- **Macbeth Color Correction**: Correction non-linéaire des couleurs

Pour plus de détails, voir [Architecture](architecture.md).

## Installation

1. Clonez ce dépôt
2. Installez les dépendances:
   ```
   pip install -r requirements.txt
   ```

## Utilisation rapide

```python
from application import Application

# Création et démarrage de l'application
app = Application()
app.run()
```

Pour des exemples plus détaillés, consultez [Exemples d'utilisation](usage_examples.md).

## Configuration

Le système utilise des fichiers de configuration thématiques dans le dossier `config/`:

- `paths_config.py`: Chemins des fichiers
- `detection_config.py`: Paramètres de détection
- `display_config.py`: Options d'affichage
- `color_config.py`: Configuration des couleurs
- `storage_config.py`: Options de stockage

Pour plus d'informations, voir [Guide de configuration](configuration.md).

## Diagrammes

- [Diagramme des composants](diagrams/component_diagram.png)
- [Diagramme de séquence](diagrams/sequence_diagram.png)
