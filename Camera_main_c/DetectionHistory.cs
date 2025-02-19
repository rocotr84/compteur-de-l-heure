using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CsvHelper;
using System.Globalization;

public class DetectionHistory
{
    private static Dictionary<int, List<string>> personDetectionHistory;
    private static StreamWriter csvOutputFile;
    private static CsvWriter csvWriter;

    public static void InitDetectionHistory(string csvFilePath)
    {
        personDetectionHistory = new Dictionary<int, List<string>>();
        
        try
        {
            csvOutputFile = new StreamWriter(csvFilePath, true);
            csvWriter = new CsvWriter(csvOutputFile, CultureInfo.InvariantCulture);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Erreur lors de l'initialisation du fichier CSV : {ex.Message}");
            throw;
        }
    }

    public static void UpdateDetectionValue(int personId, string detectedValue)
    {
        if (detectedValue != null)
        {
            if (!personDetectionHistory.ContainsKey(personId))
            {
                personDetectionHistory[personId] = new List<string>();
            }
            personDetectionHistory[personId].Add(detectedValue);
        }
    }

    public static string GetDominantDetection(int personId)
    {
        if (!personDetectionHistory.ContainsKey(personId) || !personDetectionHistory[personId].Any())
        {
            return null;
        }

        return personDetectionHistory[personId]
            .Where(v => v != null)
            .GroupBy(v => v)
            .OrderByDescending(g => g.Count())
            .Select(g => g.Key)
            .FirstOrDefault();
    }

    public static void RecordCrossing(int personId, TimeSpan currentElapsedTime)
    {
        var dominantDetection = GetDominantDetection(personId);
        if (dominantDetection != null && csvWriter != null)
        {
            var crossingTimestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            var formattedElapsedTime = $"{(int)currentElapsedTime.TotalMinutes:D2}:{currentElapsedTime.Seconds:D2}";

            try
            {
                csvWriter.WriteRecord(new string[] 
                { 
                    crossingTimestamp, 
                    formattedElapsedTime, 
                    personId.ToString(), 
                    dominantDetection 
                });
                csvWriter.NextRecord();
                csvOutputFile.Flush();
                
                Console.WriteLine($"Enregistrement CSV réussi : {crossingTimestamp}, {formattedElapsedTime}, " +
                                $"{personId}, {dominantDetection}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erreur d'enregistrement CSV : {ex.Message}");
            }
        }

        if (personDetectionHistory.ContainsKey(personId))
        {
            personDetectionHistory.Remove(personId);
        }
    }

    public static void Cleanup()
    {
        if (csvWriter != null)
        {
            csvWriter.Dispose();
        }
        if (csvOutputFile != null)
        {
            csvOutputFile.Dispose();
        }
    }
} 