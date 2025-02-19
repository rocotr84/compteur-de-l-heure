using Emgu.CV;
using Emgu.CV.Structure;
using System;
using MathNet.Numerics.Optimization;
using MathNet.Numerics.LinearAlgebra;
using Emgu.CV.CvEnum;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using MathNet.Numerics.LinearAlgebra.Double; // Pour Matrix<double>
using Vector = MathNet.Numerics.LinearAlgebra.Vector<double>;

public static class MacbethColorCorrection
{
    private static double[] lastCorrectionParams;
    private static int frameCount;

    private static Mat ApplyColorCorrection(Mat pixels, double[] correctionCoefficients)
    {
        var result = new Mat(pixels.Size, pixels.Depth, pixels.NumberOfChannels);
        
        // Extraction des coefficients
        var (a_b, b_b, c_b, d_b, gamma_b) = (correctionCoefficients[0], correctionCoefficients[1], 
            correctionCoefficients[2], correctionCoefficients[3], correctionCoefficients[4]);
        var (a_g, b_g, c_g, d_g, gamma_g) = (correctionCoefficients[5], correctionCoefficients[6], 
            correctionCoefficients[7], correctionCoefficients[8], correctionCoefficients[9]);
        var (a_r, b_r, c_r, d_r, gamma_r) = (correctionCoefficients[10], correctionCoefficients[11], 
            correctionCoefficients[12], correctionCoefficients[13], correctionCoefficients[14]);

        unsafe
        {
            byte* pixelPtr = (byte*)pixels.DataPointer;
            byte* resultPtr = (byte*)result.DataPointer;
            int width = pixels.Cols;
            int height = pixels.Rows;
            int nChan = pixels.NumberOfChannels;
            int widthStep = pixels.Step;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int idx = y * widthStep + x * nChan;
                    float B = pixelPtr[idx] / 255.0f;
                    float G = pixelPtr[idx + 1] / 255.0f;
                    float R = pixelPtr[idx + 2] / 255.0f;

                    // Calcul des combinaisons linéaires
                    double B_lin = Math.Max(a_b * B + b_b * G + c_b * R + d_b, 1e-6);
                    double G_lin = Math.Max(a_g * B + b_g * G + c_g * R + d_g, 1e-6);
                    double R_lin = Math.Max(a_r * B + b_r * G + c_r * R + d_r, 1e-6);

                    // Application de la correction gamma
                    resultPtr[idx] = (byte)(Math.Min(Math.Max(Math.Pow(B_lin, gamma_b) * 255.0, 0), 255));
                    resultPtr[idx + 1] = (byte)(Math.Min(Math.Max(Math.Pow(G_lin, gamma_g) * 255.0, 0), 255));
                    resultPtr[idx + 2] = (byte)(Math.Min(Math.Max(Math.Pow(R_lin, gamma_r) * 255.0, 0), 255));
                }
            }
        }

        return result;
    }

    private static double[] CalibrerTransformationNonLineaire(Mat colorsMeasured, Mat colorsTarget)
    {
        // Initialisation avec transformation identitaire
        var initialGuess = new double[] { 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1 };

        var solver = new NelderMeadSimplex(1e-8, 1000);

        // Création d'une fonction objectif qui calcule l'erreur
        Func<Vector<double>, double> objectiveFunction = (Vector<double> coefficients) =>
        {
            using var predicted = ApplyColorCorrection(colorsMeasured, coefficients.ToArray());
            using var diff = new Mat();
            CvInvoke.Subtract(predicted, colorsTarget, diff);
            return CvInvoke.Norm(diff, NormType.L2);
        };

        var result = solver.FindMinimum(objectiveFunction, Vector<double>.Build.DenseOfArray(initialGuess));

        return result.MinimizingPoint.ToArray();
    }

    public static Mat AppliquerCorrectionNonLineaire(Mat frameMasked, double[] correctionCoefficients)
    {
        try
        {
            using var frameNormalized = new Mat();
            frameMasked.ConvertTo(frameNormalized, DepthType.Cv32F, 1.0/255.0);
            
            var frameCorrected = ApplyColorCorrection(frameNormalized, correctionCoefficients);
            
            using var frameResult = new Mat();
            frameCorrected.ConvertTo(frameResult, DepthType.Cv8U, 255.0);
            
            return frameResult;
        }
        catch (Exception e)
        {
            Console.WriteLine($"Erreur lors de la correction des couleurs: {e.Message}");
            return frameMasked;
        }
    }

    public static Mat CorrigerImage(Mat frameMasked, string cacheFile, bool detectSquares)
    {
        try
        {
            frameCount++;
            
            if (frameCount % Config.COLOR_CORRECTION_INTERVAL == 0 || 
                lastCorrectionParams == null || detectSquares)
            {
                var colorsMeasured = MacbethDetector.GetAverageColors(frameMasked, cacheFile, detectSquares);
                var colorsTarget = Config.MACBETH_REFERENCE_COLORS;

                if (colorsMeasured.Rows != colorsTarget.Rows)
                {
                    throw new Exception("Le nombre de patchs mesurés ne correspond pas au nombre de couleurs cibles (24).");
                }

                lastCorrectionParams = CalibrerTransformationNonLineaire(colorsMeasured, colorsTarget);
                Console.WriteLine($"Recalcul des paramètres de correction (frame {frameCount}, detect_squares={detectSquares})");
            }

            return AppliquerCorrectionNonLineaire(frameMasked, lastCorrectionParams);
        }
        catch (Exception e)
        {
            Console.WriteLine($"Erreur lors de la correction des couleurs: {e.Message}");
            return frameMasked;
        }
    }

    private class ColorCorrectionObjective : IObjectiveFunction, IObjectiveFunctionEvaluation
    {
        private readonly double[] _referenceColors;
        private readonly double[] _measuredColors;
        private double _lastValue;
        private Vector _currentPoint;

        public ColorCorrectionObjective(double[] referenceColors, double[] measuredColors)
        {
            _referenceColors = referenceColors;
            _measuredColors = measuredColors;
            _currentPoint = Vector.Build.Dense(3);
        }

        void IObjectiveFunction.EvaluateAt(Vector point)
        {
            _currentPoint = point;
            _lastValue = OptimizeCorrection(point.ToArray(), _referenceColors, _measuredColors);
        }

        public double Value => _lastValue;
        public Vector Point => _currentPoint;
        public bool IsGradientSupported => false;
        public bool IsHessianSupported => false;

        // Implémentation explicite des méthodes de l'interface
        Vector<double> IObjectiveFunctionEvaluation.Gradient(Vector<double> point)
        {
            return Vector.Build.Dense(point.Count);
        }

        Matrix<double> IObjectiveFunctionEvaluation.Hessian(Vector<double> point)
        {
            return Matrix.Build.Dense(point.Count, point.Count);
        }

        public IObjectiveFunction CreateNew()
        {
            return new ColorCorrectionObjective(_referenceColors, _measuredColors);
        }

        public IObjectiveFunction Fork()
        {
            return new ColorCorrectionObjective(_referenceColors, _measuredColors);
        }
    }

    private static double OptimizeCorrection(double[] parameters, double[] referenceColors, double[] measuredColors)
    {
        double error = 0;
        for (int i = 0; i < referenceColors.Length; i += 3)
        {
            double diffR = parameters[0] * measuredColors[i] - referenceColors[i];
            double diffG = parameters[1] * measuredColors[i + 1] - referenceColors[i + 1];
            double diffB = parameters[2] * measuredColors[i + 2] - referenceColors[i + 2];
            error += diffR * diffR + diffG * diffG + diffB * diffB;
        }
        return error;
    }

    public static double[] FindCorrectionFactors(double[] referenceColors, double[] measuredColors)
    {
        var objective = new ColorCorrectionObjective(referenceColors, measuredColors);
        var solver = new NelderMeadSimplex(0.001);
        var initialGuess = Vector.Build.Dense(new[] { 1.0, 1.0, 1.0 });
        
        var result = solver.FindMinimum(objective, initialGuess);
        return result.MinimizingPoint.ToArray();
    }
} 