using System;
using System.Collections.Generic;

public static class ColorWeighting
{
    private static readonly Dictionary<string, double> ColorDetectionHistory = new();

    public static double GetColorWeight(string detectedColorName, double currentTimestamp)
    {
        if (!ColorDetectionHistory.TryGetValue(detectedColorName, out double previousDetectionTime))
        {
            return 1.0;
        }

        double timeSinceLastDetection = currentTimestamp - previousDetectionTime;
        
        if (timeSinceLastDetection < Config.MIN_TIME_BETWEEN_PASSES)
        {
            // Calcul d'une pénalité progressive basée sur le temps écoulé
            double detectionPenalty = 1.0 - (timeSinceLastDetection / Config.MIN_TIME_BETWEEN_PASSES);
            return Math.Max(Config.MIN_COLOR_WEIGHT, 1.0 - detectionPenalty);
        }
        
        return 1.0;
    }

    public static void UpdateColorTimestamp(string detectedColorName, double? detectionTimestamp = null)
    {
        ColorDetectionHistory[detectedColorName] = detectionTimestamp ?? GetCurrentTimestamp();
    }

    public static Dictionary<string, double> GetWeightedColorProbabilities(
        Dictionary<string, int> detectedColorPixels,
        double? currentTimestamp = null)
    {
        var timestamp = currentTimestamp ?? GetCurrentTimestamp();
        var weightedDetectionCounts = new Dictionary<string, double>();

        foreach (var (detectedColorName, pixelCount) in detectedColorPixels)
        {
            double colorTemporalWeight = GetColorWeight(detectedColorName, timestamp);
            weightedDetectionCounts[detectedColorName] = pixelCount * colorTemporalWeight;
        }

        return weightedDetectionCounts;
    }

    private static double GetCurrentTimestamp()
    {
        return DateTimeOffset.UtcNow.ToUnixTimeSeconds();
    }
} 