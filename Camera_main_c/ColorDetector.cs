using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq; // Pour MaxBy

public class ColorDetector
{
    private static readonly Dictionary<string, MCvScalar> VisualizationColors = new()
    {
        { "rouge_fonce", new MCvScalar(0, 0, 255) },
        { "bleu_fonce", new MCvScalar(255, 0, 0) },
        { "bleu_clair", new MCvScalar(255, 128, 0) },
        { "vert_fonce", new MCvScalar(0, 255, 0) },
        { "vert_clair", new MCvScalar(0, 255, 128) },
        { "rose", new MCvScalar(255, 0, 255) },
        { "jaune", new MCvScalar(0, 255, 255) },
        { "blanc", new MCvScalar(255, 255, 255) },
        { "noir", new MCvScalar(0, 0, 0) },
        { "inconnu", new MCvScalar(128, 128, 128) }
    };

    public static string GetDominantColor(Mat frameRaw, Rectangle detectionZone)
    {
        try
        {
            using var frameDetectionZone = new Mat(frameRaw, detectionZone);
            if (frameDetectionZone.IsEmpty)
                return "inconnu";

            using var frameHsv = new Mat();
            CvInvoke.CvtColor(frameDetectionZone, frameHsv, ColorConversion.Bgr2Hsv);

            var detectedPixelsPerColor = new Dictionary<string, int>();
            foreach (var colorName in Config.COLOR_MASKS.Keys)
            {
                var (hsvMin, hsvMax) = VideoProcessor.GetColorMask(colorName);
                using var colorMask = new Mat();
                CvInvoke.InRange(frameHsv, hsvMin, hsvMax, colorMask);

                if (colorName == "rouge_fonce")
                {
                    var (hsvMin2, hsvMax2) = VideoProcessor.GetColorMask("rouge2");
                    using var mask2 = new Mat();
                    CvInvoke.InRange(frameHsv, hsvMin2, hsvMax2, mask2);
                    CvInvoke.BitwiseOr(colorMask, mask2, colorMask);
                }

                detectedPixelsPerColor[colorName] = CvInvoke.CountNonZero(colorMask);
            }

            var weightedProbabilities = ColorWeighting.GetWeightedColorProbabilities(detectedPixelsPerColor);
            var dominantColor = weightedProbabilities.MaxBy(kvp => kvp.Value);

            if (dominantColor.Value > 0)
            {
                ColorWeighting.UpdateColorTimestamp(dominantColor.Key);
                return dominantColor.Key;
            }

            return "inconnu";
        }
        catch (Exception e)
        {
            Console.WriteLine($"Erreur lors de la détection de couleur: {e.Message}");
            return "inconnu";
        }
    }

    public static void VisualizeColor(Mat frameRaw, Rectangle detectionZone, string detectedColorName)
    {
        var rectangleColor = VisualizationColors.GetValueOrDefault(detectedColorName, VisualizationColors["inconnu"]);
        CvInvoke.Rectangle(
            frameRaw,
            detectionZone,
            rectangleColor,
            2
        );
    }

    public static (string ColorName, double Ratio, int PixelCount) DetectDominantColor(Mat frameRoi)
    {
        try
        {
            using var hsv = new Mat();
            CvInvoke.CvtColor(frameRoi, hsv, ColorConversion.Bgr2Hsv);
            
            int totalPixels = frameRoi.Rows * frameRoi.Cols;
            var detectedPixels = new Dictionary<string, (double Ratio, int Count)>();

            foreach (var colorName in Config.COLOR_RANGES.Keys)
            {
                var (hsvMin, hsvMax) = VideoProcessor.GetColorMask(colorName);
                using var mask = new Mat();
                CvInvoke.InRange(hsv, hsvMin, hsvMax, mask);

                if (colorName == "rouge")
                {
                    var (hsvMin2, hsvMax2) = VideoProcessor.GetColorMask("rouge2");
                    using var mask2 = new Mat();
                    CvInvoke.InRange(hsv, hsvMin2, hsvMax2, mask2);
                    CvInvoke.BitwiseOr(mask, mask2, mask);
                }

                int pixelCount = CvInvoke.CountNonZero(mask);
                double ratio = (double)pixelCount / totalPixels;
                detectedPixels[colorName] = (ratio, pixelCount);
            }

            var dominantColor = detectedPixels.OrderByDescending(kvp => kvp.Value.Ratio).FirstOrDefault();
            return (dominantColor.Key, dominantColor.Value.Ratio, dominantColor.Value.Count);
        }
        catch (Exception e)
        {
            Console.WriteLine($"Erreur lors de la détection des couleurs: {e.Message}");
            return (null, 0, 0);
        }
    }
} 