"""
Point d'entrée principal de l'application.

Ce module crée et exécute l'application de détection et de suivi.
"""

from application import Application
import cProfile

def main():
    """Fonction principale du programme."""
    app = Application()
    app.run()

# Point d'entrée du programme
if __name__ == "__main__":
    #cProfile.run('main()')
    main()