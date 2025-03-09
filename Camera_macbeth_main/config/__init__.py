"""
Module de configuration centralisé.

Ce module importe et expose toutes les configurations de l'application
pour faciliter leur accès depuis les autres modules.
"""

# Import de toutes les configurations
from .paths_config import *
from .detection_config import *
from .display_config import *
from .storage_config import *
from .color_config import *

# Variables globales partagées
VIDEO_OUTPUT_WRITER = None
CSV_OUTPUT_FILE = None