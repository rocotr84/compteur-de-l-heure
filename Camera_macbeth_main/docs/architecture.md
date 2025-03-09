# Architecture du système

## Vue d'ensemble

Le système est conçu selon une architecture modulaire où chaque composant a une responsabilité spécifique. La classe `Application` sert de point central pour orchestrer les interactions entre les différents modules.

## Composants principaux

### Application

La classe `Application` est le cœur du système. Elle:

- Initialise tous les composants
- Gère le cycle de vie de l'application
- Coordonne le traitement des frames
- Gère les événements et les erreurs

### Tracker

Le module de suivi est responsable de:

- Détecter les personnes dans chaque frame
- Maintenir l'identité des personnes à travers les frames
- Détecter les franchissements de ligne
- Gérer les trajectoires

### Video Processor

Ce module s'occupe de:

- Charger et prétraiter les frames vidéo
- Appliquer les masques de détection
- Coordonner la correction des couleurs

### Macbeth Color Correction

Module spécialisé dans:

- La détection de la charte Macbeth
- La correction non-linéaire des couleurs
- L'optimisation des paramètres de correction

### Color Detector

Responsable de:

- Détecter les couleurs dominantes
- Appliquer les pondérations temporelles
- Visualiser les résultats de détection

### Detection History

Gère:

- L'historique des détections par personne
- L'enregistrement des passages en CSV ou SQLite
- La détermination des valeurs dominantes

### Display Manager

S'occupe de:

- Afficher les résultats visuels
- Dessiner les éléments d'interface
- Gérer l'enregistrement vidéo

## Flux de données

1. L'application lit une frame de la vidéo
2. Le Video Processor prétraite la frame
3. Le Tracker détecte et suit les personnes
4. Le Color Detector identifie les couleurs dominantes
5. L'application détecte les franchissements de ligne
6. Les données sont enregistrées via Detection History
7. Le Display Manager affiche les résultats

## Diagramme des composants

![Diagramme des composants](diagrams/component_diagram.png)

## Diagramme de séquence

![Diagramme de séquence](diagrams/sequence_diagram.png)
