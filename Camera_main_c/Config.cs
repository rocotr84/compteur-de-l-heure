using Emgu.CV;
using System.Drawing;

public static class Config
{
    // Chemins des fichiers
    public const string VIDEO_INPUT_PATH = @"../Camera_macbeth_main/resources/video_input.mp4";
    public const string DETECTION_MASK_PATH = @"../Camera_macbeth_main/resources/detection_mask.png";
    public const string CSV_OUTPUT_PATH = @"../Camera_macbeth_main/output/detections.csv";
    public const string CACHE_FILE_PATH = @"../Camera_macbeth_main/cache/macbeth_colors.json";

    // Paramètres de détection
    public const bool DETECT_SQUARES = true;
    public const string DETECTION_MODE = "default";

    // Points de la ligne de détection
    public static readonly Point LineStart = new Point(0, 0);
    public static readonly Point LineEnd = new Point(100, 100);

    // Paramètres d'affichage
    public const int DISPLAY_WIDTH = 1920;
    public const int DISPLAY_HEIGHT = 1080;
    public const string WINDOW_NAME = "Détection de passage";

    // Paramètres de détection des couleurs
    public const double COLOR_SIMILARITY_THRESHOLD = 0.85;
    public const int MIN_CONTOUR_AREA = 100;
    public const int MAX_CONTOUR_AREA = 10000;
} 