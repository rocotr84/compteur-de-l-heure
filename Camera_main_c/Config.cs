using Emgu.CV;
using System.Drawing;
using System.Collections.Generic;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

public struct ColorMask
{
    public ScalarArray Min { get; set; }
    public ScalarArray Max { get; set; }
}

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

    // Nouveaux paramètres
    public const int DESIRED_FPS = 30;
    
    // Dictionnaires pour les masques de couleur
    public static readonly Dictionary<string, (double[] Min, double[] Max)> COLOR_RANGES = 
        new Dictionary<string, (double[] Min, double[] Max)>
    {
        { "rouge_fonce", (new double[] { 0, 100, 100 }, new double[] { 10, 255, 255 }) },
        { "rouge2", (new double[] { 170, 100, 100 }, new double[] { 180, 255, 255 }) },
        { "bleu_fonce", (new double[] { 100, 100, 100 }, new double[] { 130, 255, 255 }) },
        { "bleu_clair", (new double[] { 90, 50, 50 }, new double[] { 110, 255, 255 }) },
        { "vert_fonce", (new double[] { 60, 100, 100 }, new double[] { 80, 255, 255 }) },
        { "vert_clair", (new double[] { 40, 50, 50 }, new double[] { 80, 255, 255 }) },
        { "rose", (new double[] { 140, 50, 50 }, new double[] { 170, 255, 255 }) },
        { "jaune", (new double[] { 20, 100, 100 }, new double[] { 40, 255, 255 }) },
        { "blanc", (new double[] { 0, 0, 200 }, new double[] { 180, 30, 255 }) },
        { "noir", (new double[] { 0, 0, 0 }, new double[] { 180, 255, 30 }) }
    };

    public static readonly Dictionary<string, ColorMask> COLOR_MASKS = 
        new Dictionary<string, ColorMask>();

    // Paramètres du tracker
    public const int MAX_DISAPPEAR_FRAMES = 30;
    public const float MIN_CONFIDENCE = 0.5f;
    public const float IOU_THRESHOLD = 0.3f;
    public const string MODEL_PATH = @"../Camera_macbeth_c#/models/yolo11n.onnx";

    // Paramètres de pondération des couleurs
    public const double MIN_TIME_BETWEEN_PASSES = 5.0; // 5 secondes
    public const double MIN_COLOR_WEIGHT = 0.1;

    // Paramètres d'affichage
    public const bool SHOW_ROI_AND_COLOR = true;
    public const bool SHOW_TRAJECTORIES = true;
    public const bool SHOW_CENTER = true;
    public const bool SHOW_LABELS = true;
    public const bool SAVE_VIDEO = false;
    public const string VIDEO_OUTPUT_PATH = @"../Camera_macbeth_c#/output/output.avi";
    public const int VIDEO_FPS = 30;

    // Paramètres de correction des couleurs Macbeth
    public const int COLOR_CORRECTION_INTERVAL = 30;
    public static readonly Mat MACBETH_REFERENCE_COLORS;
    
    private static readonly byte[,] macbethColors = new byte[,]
    {
        {115, 82, 68}, // Ajoutez vos 24 couleurs Macbeth ici
        {194, 150, 130}, // Couleur 2
        {98, 122, 157}, // Couleur 3
        {87, 108, 67}, // Couleur 4
        {133, 128, 177}, // Couleur 5
        {103, 189, 170}, // Couleur 6
        {214, 126, 44}, // Couleur 7
        {80, 91, 166}, // Couleur 8
        {193, 90, 99}, // Couleur 9
        {94, 60, 108}, // Couleur 10
        {157, 188, 64}, // Couleur 11
        {224, 163, 46}, // Couleur 12
        {56, 61, 150}, // Couleur 13
        {70, 148, 73}, // Couleur 14
        {175, 54, 60}, // Couleur 15    
        {231, 199, 31}, // Couleur 16
        {187, 86, 149}, // Couleur 17
        {8, 133, 161}, // Couleur 18
        {243, 243, 242}, // Couleur 19
        {200, 200, 200}, // Couleur 20
        {160, 160, 160}, // Couleur 21
        {122, 122, 121}, // Couleur 22
        {85, 85, 85}, // Couleur 23
        {52, 52, 52} // Couleur 24
    };

    static Config()
    {
        MACBETH_REFERENCE_COLORS = new Mat(24, 1, DepthType.Cv8U, 3);
        InitializeMacbethReferenceColors();
    }

    public static void InitializeMacbethReferenceColors()
    {
        for (int i = 0; i < 24; i++)
        {
            unsafe
            {
                byte* ptr = (byte*)MACBETH_REFERENCE_COLORS.DataPointer;
                ptr[i * 3] = macbethColors[i, 0];
                ptr[i * 3 + 1] = macbethColors[i, 1];
                ptr[i * 3 + 2] = macbethColors[i, 2];
            }
        }
    }
} 