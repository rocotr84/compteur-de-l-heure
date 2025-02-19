using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text.Json;
using System.IO;
using Emgu.CV.Util;
using System.Linq;

public static class MacbethDetector
{
    private static Point[] OrderPoints(Point[] pts)
    {
        var rect = new Point[4];
        var sum = pts.Select(p => p.X + p.Y).ToArray();
        var diff = pts.Select(p => p.X - p.Y).ToArray();

        rect[0] = pts[Array.IndexOf(sum, sum.Min())];    // top-left
        rect[2] = pts[Array.IndexOf(sum, sum.Max())];    // bottom-right
        rect[1] = pts[Array.IndexOf(diff, diff.Min())];  // top-right
        rect[3] = pts[Array.IndexOf(diff, diff.Max())];  // bottom-left

        return rect;
    }

    public static (Mat WarpedFrame, Rectangle[] Squares) DetectMacbethInScene(Mat frameRaw, string cacheFile)
    {
        using var frameHsv = new Mat();
        CvInvoke.CvtColor(frameRaw, frameHsv, ColorConversion.Bgr2Hsv);

        // Détection du cadre noir
        var lowerBlack = new ScalarArray(new MCvScalar(0, 0, 0));
        var upperBlack = new ScalarArray(new MCvScalar(180, 100, 30));
        
        using var blackMask = new Mat();
        CvInvoke.InRange(frameHsv, lowerBlack, upperBlack, blackMask);

        // Nettoyage du masque
        var kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(7, 7), new Point(-1, -1));
        CvInvoke.MorphologyEx(blackMask, blackMask, MorphOp.Close, kernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar());
        CvInvoke.MorphologyEx(blackMask, blackMask, MorphOp.Open, kernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar());

        using var contours = new VectorOfVectorOfPoint();
        CvInvoke.FindContours(blackMask, contours, null, RetrType.External, ChainApproxMethod.ChainApproxSimple);

        Point[] frameCorners = null;
        double maxArea = 0;

        for (int i = 0; i < contours.Size; i++)
        {
            var contour = contours[i];
            double area = CvInvoke.ContourArea(contour);
            
            if (area > 1000)
            {
                var peri = CvInvoke.ArcLength(contour, true);
                var approx = new VectorOfPoint();
                CvInvoke.ApproxPolyDP(contour, approx, 0.02 * peri, true);

                if (approx.Size == 4 && area > maxArea)
                {
                    maxArea = area;
                    frameCorners = approx.ToArray();
                }
            }
        }

        if (frameCorners == null)
        {
            throw new Exception("Aucun grand rectangle noir détecté.");
        }

        var orderedCorners = OrderPoints(frameCorners);
        
        // Calcul des dimensions de l'image redressée
        int targetWidth = (int)Math.Max(
            Distance(orderedCorners[1], orderedCorners[0]),
            Distance(orderedCorners[2], orderedCorners[3]));
        
        int targetHeight = (int)Math.Max(
            Distance(orderedCorners[3], orderedCorners[0]),
            Distance(orderedCorners[2], orderedCorners[1]));

        var dstPoints = new PointF[]
        {
            new PointF(0, 0),
            new PointF(targetWidth - 1, 0),
            new PointF(targetWidth - 1, targetHeight - 1),
            new PointF(0, targetHeight - 1)
        };

        var perspectiveMatrix = CvInvoke.GetPerspectiveTransform(
            orderedCorners.Select(p => new PointF(p.X, p.Y)).ToArray(),
            dstPoints);

        var frameWarped = new Mat();
        CvInvoke.WarpPerspective(frameRaw, frameWarped, perspectiveMatrix, new Size(targetWidth, targetHeight));

        // Détection des carrés de couleur
        var squares = DetectColorSquares(frameWarped);

        // Sauvegarde dans le cache
        SaveToCache(cacheFile, squares, frameWarped);

        return (frameWarped, squares);
    }

    private static double Distance(Point p1, Point p2)
    {
        return Math.Sqrt(Math.Pow(p2.X - p1.X, 2) + Math.Pow(p2.Y - p1.Y, 2));
    }

    public static Mat GetAverageColors(Mat frameRaw, string cacheFile, bool detectSquares)
    {
        Mat frameWarped;
        Rectangle[] squares;

        if (!detectSquares && File.Exists(cacheFile))
        {
            var cacheData = LoadFromCache(cacheFile);
            frameWarped = new Mat(cacheData.WarpedImagePath);
            squares = cacheData.Squares;
        }
        else
        {
            (frameWarped, squares) = DetectMacbethInScene(frameRaw, cacheFile);
        }

        var colors = new Mat(24, 1, DepthType.Cv8U, 3);
        
        for (int i = 0; i < squares.Length; i++)
        {
            var roi = new Mat(frameWarped, squares[i]);
            var mean = CvInvoke.Mean(roi);
            colors.SetValue(i, 0, new byte[] { (byte)mean.V0, (byte)mean.V1, (byte)mean.V2 });
        }

        return colors;
    }

    private static void SaveToCache(string cacheFile, Rectangle[] squares, Mat warpedFrame)
    {
        var warpedImagePath = cacheFile.Replace(".json", "_warped.png");
        CvInvoke.Imwrite(warpedImagePath, warpedFrame);

        var cacheData = new
        {
            Squares = squares,
            WarpedImagePath = warpedImagePath
        };

        File.WriteAllText(cacheFile, JsonSerializer.Serialize(cacheData));
    }

    private static (Rectangle[] Squares, string WarpedImagePath) LoadFromCache(string cacheFile)
    {
        var cacheContent = File.ReadAllText(cacheFile);
        var cacheData = JsonSerializer.Deserialize<dynamic>(cacheContent);
        return (
            cacheData.GetProperty("Squares").EnumerateArray()
                .Select(s => new Rectangle(
                    s.GetProperty("X").GetInt32(),
                    s.GetProperty("Y").GetInt32(),
                    s.GetProperty("Width").GetInt32(),
                    s.GetProperty("Height").GetInt32()
                )).ToArray(),
            cacheData.GetProperty("WarpedImagePath").GetString()
        );
    }

    private static Rectangle[] DetectColorSquares(Mat frameWarped)
    {
        // Implémentation de la détection des carrés de couleur
        var squares = new List<Rectangle>();
        
        // Paramètres de détection
        int squareSize = frameWarped.Width / 6;  // Approximation pour une charte 4x6
        int spacing = squareSize / 8;
        
        // Création de la grille 4x6
        for (int row = 0; row < 4; row++)
        {
            for (int col = 0; col < 6; col++)
            {
                squares.Add(new Rectangle(
                    col * (squareSize + spacing) + spacing,
                    row * (squareSize + spacing) + spacing,
                    squareSize,
                    squareSize
                ));
            }
        }
        
        return squares.ToArray();
    }

    private static void ProcessPoints(Point[] points)
    {
        // Convertir l'expression lambda en délégué explicite
        Func<Point, int> getX = p => p.X;
        var xCoords = points.Select(getX).ToArray();
        
        Func<Point, int> getY = p => p.Y;
        var yCoords = points.Select(getY).ToArray();
    }

    private static void SetMatrixValue(Mat matrix, byte[] values, int row)
    {
        unsafe
        {
            byte* ptr = (byte*)matrix.DataPointer;
            int offset = row * 3;
            ptr[offset] = values[0];
            ptr[offset + 1] = values[1];
            ptr[offset + 2] = values[2];
        }
    }
} 