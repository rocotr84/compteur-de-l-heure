using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Dnn;
using YoloDotNet;
using YoloDotNet.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Emgu.CV.CvEnum;

public class TrackedPerson
{
    public Rectangle Bbox { get; set; }
    public int Id { get; set; }
    public float Confidence { get; set; }
    public string Value { get; set; }
    public int FramesDisappeared { get; set; }
    public List<Point> MovementTrajectory { get; set; }
    public bool HasCrossedLine { get; set; }

    public TrackedPerson(Rectangle bbox, int id, float confidence)
    {
        Bbox = bbox;
        Id = id;
        Confidence = confidence;
        Value = null;
        FramesDisappeared = 0;
        MovementTrajectory = new List<Point>();
        HasCrossedLine = false;
    }

    public void UpdateColor(Mat frame)
    {
        var detectionZone = new Rectangle(
            Bbox.X,
            Bbox.Y,
            Bbox.Width,
            Bbox.Height
        );

        Value = ColorDetector.GetDominantColor(frame, detectionZone);
    }

    public void UpdatePosition(Rectangle newBbox, float newConfidence)
    {
        Bbox = newBbox;
        Confidence = newConfidence;
        var bottomCenter = GetBboxBottomCenter(newBbox);
        MovementTrajectory.Add(bottomCenter);
        
        if (MovementTrajectory.Count > 30)
        {
            MovementTrajectory.RemoveAt(0);
        }
    }

    private Point GetBboxBottomCenter(Rectangle bbox)
    {
        return new Point(
            bbox.X + bbox.Width / 2,
            bbox.Y + bbox.Height
        );
    }
}

public class Tracker
{
    private InferenceSession _session;
    private readonly Dictionary<int, TrackedPerson> _activeTrackedPersons;
    private readonly HashSet<int> _personsCrossedLine;
    private readonly Dictionary<string, int> _lineCrossingCounter;
    private int _nextPersonId;
    private Mat _processedFrame;
    private Mat _model;

    public Tracker()
    {
        _session = new InferenceSession(Config.MODEL_PATH);
        _activeTrackedPersons = new Dictionary<int, TrackedPerson>();
        _personsCrossedLine = new HashSet<int>();
        _lineCrossingCounter = new Dictionary<string, int>();
        _nextPersonId = 0;
        _processedFrame = new Mat();
        _model = new Mat();
    }

    public List<TrackedPerson> UpdateTracking(Mat frame)
    {
        try
        {
            // Détection avec YoloDotNet
            var predictions = _model.Predict(frame.ToBitmap());
            var detections = predictions
                .Where(p => p.Label.Name == "person" && p.Confidence >= Config.MIN_CONFIDENCE)
                .Select(p => new
                {
                    Bbox = new Rectangle(
                        (int)p.Bounds.X,
                        (int)p.Bounds.Y,
                        (int)p.Bounds.Width,
                        (int)p.Bounds.Height
                    ),
                    Confidence = p.Confidence
                })
                .ToList();

            // Mise à jour du tracking
            var matchedIds = new HashSet<int>();
            var newDetections = new List<(Rectangle Bbox, float Confidence)>();

            // Association des détections aux personnes existantes
            foreach (var detection in detections)
            {
                bool matched = false;
                foreach (var person in _activeTrackedPersons.Values)
                {
                    if (!matchedIds.Contains(person.Id))
                    {
                        float iou = CalculateIoU(detection.Bbox, person.Bbox);
                        if (iou > Config.IOU_THRESHOLD)
                        {
                            person.UpdatePosition(detection.Bbox, detection.Confidence);
                            matchedIds.Add(person.Id);
                            matched = true;
                            break;
                        }
                    }
                }

                if (!matched)
                {
                    newDetections.Add((detection.Bbox, detection.Confidence));
                }
            }

            // Création de nouvelles personnes pour les détections non associées
            foreach (var detection in newDetections)
            {
                var newPerson = new TrackedPerson(detection.Bbox, _nextPersonId++, detection.Confidence);
                _activeTrackedPersons[newPerson.Id] = newPerson;
            }

            // Mise à jour des personnes non détectées
            var personIdsToRemove = new List<int>();
            foreach (var person in _activeTrackedPersons.Values)
            {
                if (!matchedIds.Contains(person.Id))
                {
                    person.FramesDisappeared++;
                    if (person.FramesDisappeared > Config.MAX_DISAPPEAR_FRAMES)
                    {
                        personIdsToRemove.Add(person.Id);
                    }
                }
            }

            // Suppression des personnes disparues
            foreach (var id in personIdsToRemove)
            {
                _activeTrackedPersons.Remove(id);
            }

            return _activeTrackedPersons.Values.ToList();
        }
        catch (Exception e)
        {
            Console.WriteLine($"Erreur lors du tracking: {e.Message}");
            return new List<TrackedPerson>();
        }
    }

    private float CalculateIoU(Rectangle box1, Rectangle box2)
    {
        int intersectionX1 = Math.Max(box1.X, box2.X);
        int intersectionY1 = Math.Max(box1.Y, box2.Y);
        int intersectionX2 = Math.Min(box1.X + box1.Width, box2.X + box2.Width);
        int intersectionY2 = Math.Min(box1.Y + box1.Height, box2.Y + box2.Height);

        if (intersectionX2 < intersectionX1 || intersectionY2 < intersectionY1)
            return 0;

        int intersectionArea = (intersectionX2 - intersectionX1) * (intersectionY2 - intersectionY1);
        int box1Area = box1.Width * box1.Height;
        int box2Area = box2.Width * box2.Height;
        int unionArea = box1Area + box2Area - intersectionArea;

        return (float)intersectionArea / unionArea;
    }

    public void MarkPersonAsCrossed(int personId)
    {
        _personsCrossedLine.Add(personId);
        _activeTrackedPersons.Remove(personId);
    }

    public bool IsPersonActive(int personId)
    {
        return _activeTrackedPersons.ContainsKey(personId);
    }

    public Dictionary<string, int> LineCrossingCounter => _lineCrossingCounter;

    public void Update(Mat frame)
    {
        frame.CopyTo(_processedFrame);
    }

    public void CheckLineCrossing()
    {
        // Logique de vérification
    }

    public Image ProcessFrame(Mat frame)
    {
        using (Mat temp = frame.Clone())
        {
            return temp.Bitmap;
        }
    }

    public void Dispose()
    {
        _session?.Dispose();
        _processedFrame?.Dispose();
        _model?.Dispose();
    }
} 