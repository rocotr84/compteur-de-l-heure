# Méthodes de Correction des Couleurs Basées sur la Charte Macbeth

Ce document présente deux approches pour corriger les couleurs d'une image en utilisant la charte Macbeth :

- **Méthode linéaire**
- **Méthode non linéaire**

---

## 1. Vue d'ensemble

La charte Macbeth contient 24 patchs de couleurs standard aux valeurs cibles connues. En utilisant ces patchs, il est possible de calculer une transformation permettant de corriger les couleurs mesurées par vos caméras afin de se rapprocher des valeurs cibles. Deux approches principales se distinguent :

- **La méthode linéaire** qui applique une transformation affine (ou linéaire) aux couleurs mesurées.
- **La méthode non linéaire** qui introduit des éléments de puissance pour gérer les non-linéarités présentes dans la réponse des capteurs.

---

## 2. Méthode Linéaire

### Principe

La méthode linéaire modélise la correction pour chaque canal (R, G, B) par une transformation affine :

**R_corrigé = a_R · R + b_R · G + c_R · B + d_R**  
  **G_corrigé = a_G · R + b_G · G + c_G · B + d_G**  
  **B_corrigé = a_B · R + b_B · G + c_B · B + d_B**

Les coefficients de cette transformation sont déterminés par une régression (méthode des moindres carrés) sur les 24 patchs de la charte Macbeth.

### Avantages

- **Simplicité** : Facile à implémenter et à comprendre.
- **Rapidité** : Calcul rapide et adapté aux applications en temps réel, à condition que l'échantillonnage de la charte couvre bien la plage de valeurs d'entrée.

### Limites et Exemples

- **Gamme de Calibration et Extrapolation** :  
  La transformation est optimale uniquement pour la plage de valeurs utilisée lors de la calibration.  
  **Exemple** : Si la calibration s’effectue sur des valeurs proches de 205 pour chaque canal RGB et que l’on traite une image ayant des valeurs autour de 105, le modèle procède à une extrapolation qui peut entraîner une correction inexacte (sous-correction ou erreur de teinte).

- **Différences de Capteurs** :  
  Même après correction, deux caméras ayant des réponses différentes lors de la capture (par exemple, un rouge mesuré à 171, 12, 26 pour une caméra et 80, 10, 12 pour l'autre) peuvent ne pas converger parfaitement vers la même valeur. La transformation linéaire minimise l'erreur globale mais ne compense pas toujours les variations intrinsèques entre capteurs.

---

## 3. Méthode Non Linéaire

### Principe

Pour prendre en compte les réponses non linéaires des capteurs, on peut appliquer une transformation du type :

**couleur_corrigée_j = (a_j · R + b_j · G + c_j · B + d_j) ^ gamma_j**

pour chaque canal j ∈ {R, G, B}.  
Les 15 paramètres (5 par canal : a, b, c, d, gamma) sont déterminés via une optimisation non linéaire (par exemple avec la fonction `least_squares` de SciPy).

### Avantages

- **Gestion des Non-linéarités** : Permet de mieux capturer les déformations de la réponse du capteur, ainsi que les effets liés à l'éclairage ou à des conditions environnementales variables.
- **Adaptabilité** : En théorie, cette approche corrige plus précisément les différences de réponses entre différents capteurs.

### Limites et Exemples

- **Complexité Computationnelle** :  
  L’optimisation non linéaire est plus coûteuse en temps de calcul et peut nécessiter une bonne initialisation des paramètres pour assurer la convergence.  
  **Exemple** : Dans certains cas, si les valeurs initiales ne sont pas judicieusement choisies, l’optimisation peut converger vers une solution locale non optimale, conduisant à une correction imprécise.

- **Sensibilité à la Plage de Calibration** :  
  De manière similaire à la méthode linéaire, si les valeurs d'entrée sont en dehors de la plage couverte par la calibration (par exemple, une valeur 105 alors que la calibration repose sur des valeurs autour de 205), la correction, même non linéaire, peut présenter des erreurs. Bien que le modèle non linéaire soit plus flexible, il reste fortement dépendant de la qualité et de la représentativité des données de calibration.

---

## 4. Calibration Durant la Course

Lors d'une course en extérieur au printemps, où le soleil est présent mais la lumière varie au cours du temps, il est crucial d'adapter la calibration pour obtenir une correction couleur fiable.

### Stratégies de Calibration

- **Calibration Initiale** :  
  Réaliser une première calibration avant le départ dans des conditions représentatives (exposition correcte, éclairage stable) permet d'établir une base de correction.

- **Calibration Dynamique** :

  - **Charte de Référence Mobile** : Vous pouvez positionner une petite charte dans le champ de vision de la caméra pour capturer périodiquement une référence connue.
  - **Mesures Périodiques** : Acquérir des images de la charte à différents moments de la course permet ensuite d'interpoler ou d'ajuster la correction en post-traitement en fonction de l'évolution de l'éclairage.

- **Enregistrement des Conditions d'Éclairage** :  
  Il peut être utile de mesurer des paramètres complémentaires, tels que la température de couleur et l'intensité lumineuse, afin d'adapter la correction en fonction des variations réelles pendant la course.

### Température de Couleur

La température de couleur décrit la teinte de la lumière ambiante et s'exprime en Kelvin (K).

- **Définition** : Elle caractérise si la lumière est plutôt "chaude" (teinte jaune/orange, environ 3000 K à 4000 K, comme la lumière tungsten) ou "froide" (teinte bleue, environ 5500 K à 6500 K, comme la lumière du jour).
- **Mesure** : Elle peut être mesurée à l'aide de capteurs spécialisés ou de spectromètres.  
  Certains appareils photo ou stations météorologiques intègrent des capteurs qui, associés à un luxmètre ou un capteur spectro, peuvent fournir cette information.

### Intensité Lumineuse

L'intensité lumineuse représente la quantité de lumière par unité de surface et est généralement exprimée en lux.

- **Définition** : Le lux indique le niveau de luminosité ambiante. Par exemple, en plein jour on peut mesurer plusieurs dizaines de milliers de lux, tandis qu'en intérieur, cela se situe souvent entre 100 et 500 lux.
- **Mesure** : Un luxmètre est généralement utilisé pour mesurer l'intensité lumineuse ambiante. Ces dispositifs sont couramment utilisés pour calibrer des installations d'éclairage ou pour mesurer l'exposition lors de prises de vue.

### Choix du Modèle de Calibration pour la Course

Pour une course en extérieur avec des variations d'éclairage :

- **Méthode Linéaire** :

  - **Avantages** : Plus simple et rapide en termes de calcul, ce qui est très utile pour un traitement en temps réel.
  - **Limites** : Peut ne pas capturer toutes les variations non linéaires provenant de changements rapides de l'environnement lumineux.

- **Méthode Non Linéaire** :
  - **Avantages** : Fournit une meilleure adaptation en modélisant précisément les non-linéarités dues aux variations d'éclairage.
  - **Limites** : Plus gourmande en ressources de calcul et nécessite une bonne initialisation, ce qui peut être un inconvénient en temps réel.

**Recommandation** :

- Si le temps de traitement est critique et que les variations d'éclairage restent modérées, la calibration linéaire peut être suffisante.
- En revanche, si la précision de correction est primordiale et que vous pouvez vous permettre une charge de calcul plus élevée (ou effectuer le traitement en post-course), la calibration non linéaire est à privilégier pour mieux compenser les variations d'illumination.

---

## 5. Conclusion

La calibration par la charte Macbeth est un moyen efficace de standardiser les couleurs entre différents capteurs. Toutefois, le choix de la méthode (linéaire ou non linéaire) et l'approche adoptée (calibration initiale versus calibration dynamique) doivent être réfléchis en fonction des conditions réelles d'utilisation. En extérieur, avec une lumière changeante pendant la course, il est recommandé d'envisager une calibration dynamique et de prendre en compte des mesures de température de couleur et d'intensité lumineuse afin d'ajuster au mieux la correction couleur.

---
