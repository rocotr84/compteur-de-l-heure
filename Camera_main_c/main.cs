using System;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using System.Threading.Tasks;
using System.Collections.Generic;
using Emgu.CV.Dnn;
using Emgu.CV.Cuda;
using static Emgu.CV.Dnn.Backend;

public class Program
{
    private static VideoCapture _capture;
    private static DisplayManager _displayManager;
    private static VideoProcessor _videoProcessor;
    private static Tracker _tracker;

    public static void Main()
    {
        Initialize();
        ProcessVideo();
        Cleanup();
    }

    private static void Initialize()
    {
        _capture = new VideoCapture(0); // ou le chemin de votre vidéo
        _displayManager = new DisplayManager(_capture);
        _videoProcessor = new VideoProcessor();
        _tracker = new Tracker();

        if (CudaInvoke.HasCuda)
        {
            try
            {
                CvInvoke.UseOpenCL = false;
                CudaInvoke.PrintCudaDeviceInfo(CudaInvoke.GetDevice());
            }
            catch (Exception ex)
            {
                Console.WriteLine($"CUDA initialization error: {ex.Message}");
            }
        }
    }

    private static void ProcessVideo()
    {
        Mat frame = new Mat();
        while (_capture.IsOpened && !Console.KeyAvailable)
        {
            _capture.Read(frame);
            if (frame.IsEmpty) break;

            _videoProcessor.ProcessFrame(frame);
            _tracker.Update(frame);
            _displayManager.ShowFrame(frame);

            if (CvInvoke.WaitKey(1) == 27) break; // Échap pour quitter
        }
        frame.Dispose();
    }

    private static void Cleanup()
    {
        _capture?.Dispose();
        _videoProcessor?.Dispose();
        _displayManager?.ReleaseDisplay();
    }
}