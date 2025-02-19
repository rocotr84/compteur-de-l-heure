using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Drawing;

public class DisplayManager
{
    private const string DisplayWindowName = "Tracking";
    private static VideoWriter _videoOutputWriter;
    private static readonly DateTime DisplayStartTime = DateTime.Now;
    private Mat _display;
    private VideoCapture _capture;
    private const int WINDOW_WIDTH = 1280;
    private const int WINDOW_HEIGHT = 720;

    public DisplayManager(VideoCapture capture)
    {
        _capture = capture;
        _display = new Mat();
        InitDisplay();
    }

    public void InitDisplay()
    {
        if (Config.SAVE_VIDEO)
        {
            int fourcc = VideoWriter.Fourcc('M', 'J', 'P', 'G');
            _videoOutputWriter = new VideoWriter(
                Config.VIDEO_OUTPUT_PATH,
                fourcc,
                Config.VIDEO_FPS,
                new Size(Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT),
                true
            );
        }
        CvInvoke.NamedWindow("Display", WindowFlags.Normal);
        CvInvoke.ResizeWindow("Display", WINDOW_WIDTH, WINDOW_HEIGHT);
    }

    public void DrawPerson(Mat frameDisplay, TrackedPerson trackedPerson)
    {
        var bbox = trackedPerson.Bbox;
        CvInvoke.Rectangle(
            frameDisplay,
            bbox,
            new MCvScalar(0, 255, 0),
            2
        );

        if (Config.SHOW_ROI_AND_COLOR)
        {
            var detectionZone = new Rectangle(
                bbox.X + (int)(bbox.Width * 0.30),
                bbox.Y + (int)(bbox.Height * 0.2),
                (int)(bbox.Width * 0.40),
                (int)(bbox.Height * 0.2)
            );

            if (IsRectangleValid(detectionZone, frameDisplay))
            {
                var detectedValue = ColorDetector.GetDominantColor(frameDisplay, detectionZone);
                trackedPerson.Value = detectedValue;
                ColorDetector.VisualizeColor(frameDisplay, detectionZone, detectedValue);
            }
        }

        if (Config.SHOW_LABELS)
        {
            DrawPersonLabel(frameDisplay, trackedPerson, bbox.X, bbox.Y);
        }
        if (Config.SHOW_TRAJECTORIES)
        {
            DrawPersonTrajectory(frameDisplay, trackedPerson);
        }
        if (Config.SHOW_CENTER)
        {
            DrawPersonCenter(frameDisplay, trackedPerson);
        }
    }

    private void DrawPersonLabel(Mat frameDisplay, TrackedPerson person, int labelPosX, int labelPosY)
    {
        string labelText = person.Id.ToString();

        if (Config.DETECTION_MODE == "color" && person.Value != null)
        {
            var colorDisplayNames = new Dictionary<string, string>
            {
                { "rouge_fonce", "rouge fonce" },
                { "bleu_fonce", "bleu fonce" },
                { "bleu_clair", "bleu clair" },
                { "vert_fonce", "vert fonce" },
                { "vert_clair", "vert clair" },
                { "rose", "rose" },
                { "jaune", "jaune" },
                { "blanc", "blanc" },
                { "noir", "noir" },
                { "inconnu", "inconnu" }
            };

            string displayColorName = colorDisplayNames.GetValueOrDefault(person.Value, person.Value);
            labelText = $"{person.Id} - {displayColorName}";
        }
        else if (Config.DETECTION_MODE == "number" && person.Value != null)
        {
            labelText = $"{person.Id} - N°{person.Value}";
        }

        DrawLabel(frameDisplay, labelText, new Point(labelPosX, labelPosY - 5));
    }

    private void DrawPersonTrajectory(Mat frameDisplay, TrackedPerson person)
    {
        if (Config.SHOW_TRAJECTORIES && person.MovementTrajectory.Count > 1)
        {
            for (int i = 0; i < person.MovementTrajectory.Count - 1; i++)
            {
                CvInvoke.Line(
                    frameDisplay,
                    person.MovementTrajectory[i],
                    person.MovementTrajectory[i + 1],
                    new MCvScalar(0, 0, 255),
                    2
                );
            }
        }
    }

    private void DrawPersonCenter(Mat frameDisplay, TrackedPerson person)
    {
        var centerX = person.Bbox.X + person.Bbox.Width / 2;
        var centerY = person.Bbox.Bottom;
        CvInvoke.Circle(
            frameDisplay,
            new Point(centerX, centerY),
            1,
            new MCvScalar(0, 0, 255),
            -1
        );
    }

    public void DrawCounters(Mat frameDisplay, Dictionary<string, int> counterValues)
    {
        int textYPosition = 30;
        foreach (var (valueName, count) in counterValues)
        {
            CvInvoke.PutText(
                frameDisplay,
                $"{valueName}: {count}",
                new Point(10, textYPosition),
                FontFace.HersheySimplex,
                1.0,
                new MCvScalar(255, 255, 255),
                2
            );
            textYPosition += 30;
        }
    }

    public void DrawCrossingLine(Mat frameDisplay, Point lineStartPoint, Point lineEndPoint)
    {
        CvInvoke.Line(
            frameDisplay,
            lineStartPoint,
            lineEndPoint,
            new MCvScalar(0, 0, 255),
            2
        );
    }

    public TimeSpan DrawTimer(Mat frameDisplay)
    {
        var elapsedTime = DateTime.Now - DisplayStartTime;
        string timerText = $"{(int)elapsedTime.TotalMinutes:D2}:{elapsedTime.Seconds:D2}";

        DrawLabel(frameDisplay, timerText, new Point(10, 30));

        return elapsedTime;
    }

    public (bool ShouldQuit, TimeSpan ElapsedTime) ShowFrame(Mat frameDisplay)
    {
        var elapsedTime = DrawTimer(frameDisplay);

        if (Config.SAVE_VIDEO && _videoOutputWriter != null)
        {
            _videoOutputWriter.Write(frameDisplay);
            return (false, elapsedTime);
        }

        frameDisplay.CopyTo(_display);
        CvInvoke.Imshow("Display", _display);
        return ((CvInvoke.WaitKey(1) & 0xFF) == 'q', elapsedTime);
    }

    private bool IsRectangleValid(Rectangle rect, Mat frame)
    {
        return rect.X >= 0 && rect.Y >= 0 && 
               rect.Right <= frame.Width && rect.Bottom <= frame.Height;
    }

    public void ReleaseDisplay()
    {
        _videoOutputWriter?.Dispose();
        _display.Dispose();
        CvInvoke.DestroyAllWindows();
    }

    public void DrawLabel(Mat frameDisplay, string labelText, Point labelPos)
    {
        CvInvoke.PutText(
            frameDisplay,
            labelText,
            new Point(labelPos.X, labelPos.Y - 5),
            FontFace.HersheySimplex,
            1.0,
            new MCvScalar(255, 255, 255),
            2
        );
    }
} 