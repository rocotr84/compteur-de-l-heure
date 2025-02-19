using System;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using System.Collections.Generic;
using System.Runtime.Caching;
using System.IO;

public class VideoProcessor
{
    private static Mat mask;
    private static Mat resizedMask;
    private static int frameCount;
    private static double[] lastCorrectionCoefficients;
    private static MemoryCache resizeMatrixCache = MemoryCache.Default;
    private Mat _frame;
    private Mat _hsv;
    private Mat _mask;
    private readonly int[] _hsvMin = new int[] { 0, 0, 0 };
    private readonly int[] _hsvMax = new int[] { 180, 255, 255 };

    public VideoProcessor()
    {
        _frame = new Mat();
        _hsv = new Mat();
        _mask = new Mat();
    }

    public static void LoadMask(string maskPath)
    {
        Console.WriteLine($"Tentative de chargement du masque depuis: {maskPath}");

        try
        {
            if (!File.Exists(maskPath))
            {
                Console.WriteLine($"ERREUR: Le fichier de masque n'existe pas: {maskPath}");
                CreateDefaultMask();
                return;
            }

            mask = CvInvoke.Imread(maskPath, ImreadModes.Grayscale);
            
            if (mask.IsEmpty)
            {
                Console.WriteLine($"ERREUR: Impossible de charger le masque: {maskPath}");
                CreateDefaultMask();
            }
            else
            {
                var originalShape = mask.Size;
                resizedMask = new Mat();
                CvInvoke.Resize(mask, resizedMask, new Size(Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT));
                Console.WriteLine($"Masque chargé avec succès: {originalShape} et redimensionné à {(Config.DISPLAY_HEIGHT, Config.DISPLAY_WIDTH)}");
            }
        }
        catch (Exception e)
        {
            Console.WriteLine($"ERREUR lors du chargement du masque: {e.Message}");
            CreateDefaultMask();
        }
    }

    private static void CreateDefaultMask()
    {
        if (Config.DISPLAY_WIDTH <= 0 || Config.DISPLAY_HEIGHT <= 0)
        {
            throw new InvalidOperationException("Les dimensions de sortie doivent être initialisées");
        }

        mask = new Mat(Config.DISPLAY_HEIGHT, Config.DISPLAY_WIDTH, DepthType.Cv8U, 1);
        mask.SetTo(new MCvScalar(255));
        resizedMask = mask.Clone();
        Console.WriteLine("Création d'un masque blanc par défaut");
    }

    public static VideoCapture SetupVideoCapture(string videoPath)
    {
        var capture = new VideoCapture(videoPath);
        if (!capture.IsOpened)
        {
            throw new InvalidOperationException("Erreur : Impossible d'accéder à la vidéo.");
        }

        capture.Set(CapProp.Fps, Config.DESIRED_FPS);
        return capture;
    }

    private static Mat ApplyMask(Mat frame, Mat mask)
    {
        var result = new Mat();
        CvInvoke.BitwiseAnd(frame, frame, result, mask);
        return result;
    }

    private static Mat GetResizeMatrix(Size inputShape)
    {
        string cacheKey = $"resize_matrix_{inputShape.Width}_{inputShape.Height}";
        
        if (resizeMatrixCache.Contains(cacheKey))
        {
            return (Mat)resizeMatrixCache.Get(cacheKey);
        }

        if (inputShape.Width == Config.DISPLAY_WIDTH && inputShape.Height == Config.DISPLAY_HEIGHT)
        {
            return null;
        }

        var srcPoints = new PointF[]
        {
            new PointF(0, 0),
            new PointF(inputShape.Width - 1, 0),
            new PointF(0, inputShape.Height - 1)
        };

        var dstPoints = new PointF[]
        {
            new PointF(0, 0),
            new PointF(Config.DISPLAY_WIDTH - 1, 0),
            new PointF(0, Config.DISPLAY_HEIGHT - 1)
        };

        var matrix = CvInvoke.GetAffineTransform(srcPoints, dstPoints);
        
        var cacheItem = new CacheItem(cacheKey, matrix);
        var policy = new CacheItemPolicy { SlidingExpiration = TimeSpan.FromMinutes(10) };
        resizeMatrixCache.Add(cacheItem, policy);

        return matrix;
    }

    public static Mat ProcessFrame(Mat frameRaw, string cacheFile, bool detectSquares)
    {
        try
        {
            Mat frameResized;
            var matrix = GetResizeMatrix(frameRaw.Size);
            
            if (matrix == null)
            {
                frameResized = frameRaw.Clone();
            }
            else
            {
                frameResized = new Mat();
                CvInvoke.WarpAffine(frameRaw, frameResized, matrix, 
                    new Size(Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT));
            }

            Mat frameMasked;
            if (resizedMask != null)
            {
                frameMasked = ApplyMask(frameResized, resizedMask);
            }
            else
            {
                frameMasked = frameResized;
            }

            var frameCorrected = MacbethColorCorrection.CorrigerImage(frameMasked, cacheFile, detectSquares);
            if (frameCorrected == null)
            {
                Console.WriteLine("Erreur: La correction des couleurs a échoué");
                return frameMasked;
            }

            return frameCorrected;
        }
        catch (Exception e)
        {
            Console.WriteLine($"Erreur lors du traitement de la frame: {e.Message}");
            return frameRaw;
        }
    }

    public static void InitializeColorMasks()
    {
        try
        {
            foreach (var colorRange in Config.COLOR_RANGES)
            {
                var (hsvMin, hsvMax) = colorRange.Value;
                
                if (hsvMin[0] < 0 || hsvMax[0] > 180 ||
                    hsvMin[1] < 0 || hsvMax[1] > 255 ||
                    hsvMin[2] < 0 || hsvMax[2] > 255)
                {
                    throw new ArgumentException($"Valeurs HSV invalides pour {colorRange.Key}");
                }

                Config.COLOR_MASKS[colorRange.Key] = new ColorMask
                {
                    Min = new ScalarArray(hsvMin),
                    Max = new ScalarArray(hsvMax)
                };
            }

            Console.WriteLine($"Masques de couleurs initialisés pour {Config.COLOR_MASKS.Count} couleurs");
        }
        catch (Exception e)
        {
            Console.WriteLine($"Erreur lors de l'initialisation des masques de couleurs: {e.Message}");
            throw;
        }
    }

    public static (ScalarArray Min, ScalarArray Max) GetColorMask(string colorName)
    {
        if (!Config.COLOR_MASKS.ContainsKey(colorName))
        {
            throw new ArgumentException($"Couleur non reconnue : {colorName}");
        }
        return (Config.COLOR_MASKS[colorName].Min, Config.COLOR_MASKS[colorName].Max);
    }

    public void ProcessFrame(Mat frame)
    {
        frame.CopyTo(_frame);
        CvInvoke.CvtColor(_frame, _hsv, ColorConversion.Bgr2Hsv);
        
        var correctedMin = new MCvScalar(_hsvMin[0], _hsvMin[1], _hsvMin[2], 0);
        var correctedMax = new MCvScalar(_hsvMax[0], _hsvMax[1], _hsvMax[2], 0);

        CvInvoke.InRange(_hsv, correctedMin, correctedMax, _mask);
    }

    public void Dispose()
    {
        _frame?.Dispose();
        _hsv?.Dispose();
        _mask?.Dispose();
    }
} 