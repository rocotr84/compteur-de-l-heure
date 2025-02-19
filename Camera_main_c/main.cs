using Emgu.CV;
using System;
using System.Threading.Tasks;

public class Program
{
    private static VideoCapture videoCapture;
    private static Dictionary<string, int> lineCrossingCounter;
    private static bool isRunning = true;

    public static void Main(string[] args)
    {
        try
        {
            InitializeSystem();
            ProcessVideo();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Erreur dans la boucle principale : {ex.Message}");
        }
        finally
        {
            Cleanup();
        }
    }

    private static void InitializeSystem()
    {
        Console.CancelKeyPress += (s, e) =>
        {
            Console.WriteLine("\nSauvegarde des données et arrêt du programme...");
            Cleanup();
            Environment.Exit(0);
        };

        InitDetectionHistory(CSV_OUTPUT_PATH);
        Console.WriteLine($"Démarrage du suivi en mode {DETECTION_MODE}...");

        // Initialisation des masques de couleurs
        InitializeColorMasks();
        Console.WriteLine("Masques de couleurs initialisés avec succès");

        // Configuration de la capture vidéo
        videoCapture = SetupVideoCapture(VIDEO_INPUT_PATH);

        // Chargement du masque de détection
        LoadMask(DETECTION_MASK_PATH);
        Console.WriteLine("Masque de détection chargé et pré-calculé");

        // Initialisation du tracker et de l'affichage
        trackerState = CreateTracker();
        InitDisplay();

        // Configuration du dispositif de calcul
        var device = SetupDevice();
        trackerState.PersonDetectionModel = trackerState.PersonDetectionModel.To(device);

        // Détection initiale des couleurs Macbeth
        using (var initialFrame = new Mat())
        {
            if (videoCapture.Read(initialFrame))
            {
                try
                {
                    GetAverageColors(initialFrame, CACHE_FILE_PATH, true);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Erreur lors de la détection des couleurs Macbeth: {ex.Message}");
                }
            }
        }
    }

    private static void ProcessVideo()
    {
        using (var frame = new Mat())
        {
            while (isRunning)
            {
                if (!videoCapture.Read(frame))
                    break;

                var processedFrame = ProcessFrame(frame, CACHE_FILE_PATH, DETECT_SQUARES);
                var trackedPersons = UpdateTracker(trackerState, processedFrame);

                foreach (var person in trackedPersons)
                {
                    if (person.Value != null)
                    {
                        UpdateDetectionValue(person.Id, person.Value);
                    }

                    if (CheckLineCrossing(person, lineStart, lineEnd))
                    {
                        HandleLineCrossing(person.Id, processedFrame);
                    }

                    DrawPerson(processedFrame, person);
                }

                DrawCrossingLine(processedFrame, lineStart, lineEnd);
                DrawCounters(processedFrame, trackerState.LineCrossingCounter);

                if (ShowFrame(processedFrame))
                    break;
            }
        }
    }

    private static void HandleLineCrossing(int personId, Mat frame)
    {
        if (trackerState.ActiveTrackedPersons.Contains(personId))
        {
            Console.WriteLine($"!!! Ligne traversée par ID={personId} !!!");
            var currentTime = DrawTimer(frame);

            var detectedValue = GetDominantDetection(personId);
            UpdateDetectionValue(personId, detectedValue);

            RecordCrossing(personId, currentTime);
            var dominantValue = GetDominantDetection(personId);
            
            if (dominantValue != null)
            {
                trackerState.LineCrossingCounter[dominantValue]++;
            }

            MarkPersonAsCrossed(trackerState, personId);
            Console.WriteLine($"Personne ID={personId} marquée comme ayant traversé");
        }
    }

    private static Device SetupDevice()
    {
        if (CvInvoke.HaveOpenCL)
        {
            CvInvoke.UseOpenCL = true;
            Console.WriteLine("OpenCL activé");
            return Device.GPU;
        }
        else
        {
            Console.WriteLine("OpenCL non disponible, utilisation du CPU");
            return Device.CPU;
        }
    }

    private static void Cleanup()
    {
        Console.WriteLine("Fermeture du programme...");
        CleanupDetectionHistory();
        videoCapture?.Dispose();
        ReleaseDisplay();
    }
}