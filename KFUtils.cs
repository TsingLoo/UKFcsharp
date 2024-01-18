using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

public static class KFUtils
{
    public static Tuple<Matrix<float>, Vector<float>, Vector<float>> ScaledSigmaPoints(Vector<float> x, Matrix<float> P, float alpha = 0.001f, float beta = 2f, float kappa = 0f)
    {
        int n = x.Count;
        float lambda_ = alpha * alpha * (n + kappa) - n;

        // TODO: Whether (n + lambda) * P or the code below
        
        //Debug.Log(((n + lambda_) * P));


        //Debug.Log($"[{nameof(UKFTools)}] kappa is {kappa}");
        
        var S = (2) * P .Cholesky().Factor;
        
        // Generate sigma points
        Matrix<float> X = Matrix<float>.Build.Dense(2 * n + 1, n);
        X.SetRow(0, x);
        for (int i = 0; i < n; i++)
        {
            X.SetRow(i + 1, x + S.Row(i));
            X.SetRow(n + i + 1, x - S.Row(i));
        }

        // Compute weights
        Vector<float> W_m = Vector<float>.Build.Dense(2 * n + 1, 1f / (2f * (n + lambda_)));
        Vector<float> W_c = Vector<float>.Build.Dense(2 * n + 1, 1f / (2f * (n + lambda_)));
        W_m[0] = lambda_ / (n + lambda_);
        W_c[0] = lambda_ / (n + lambda_) + (1 - alpha * alpha + beta);

        return new Tuple<Matrix<float>, Vector<float>, Vector<float>>(X, W_m, W_c);
    }
    
    public static Matrix<float> Q_DiscreteWhiteNoise(int dim, float dt, float var = 1.0f)
    {
        if (dim < 2 || dim > 4)
        {
            throw new ArgumentException("dim must be between 2 and 4");
        }

        Matrix<float> Q;
        switch (dim)
        {
            case 2:
                Q = Matrix<float>.Build.DenseOfArray(new float[,] {
                    { 0.25f * dt * dt * dt * dt, 0.5f * dt * dt * dt },
                    { 0.5f * dt * dt * dt, dt * dt }
                });
                break;
            case 3:
                Q = Matrix<float>.Build.DenseOfArray(new float[,] {
                    { 0.25f * dt * dt * dt * dt, 0.5f * dt * dt * dt, 0.5f * dt * dt },
                    { 0.5f * dt * dt * dt, dt * dt, dt },
                    { 0.5f * dt * dt, dt, 1 }
                });
                break;
            default: // dim == 4
                Q = Matrix<float>.Build.DenseOfArray(new float[,] {
                    { dt * dt * dt * dt * dt * dt / 36, dt * dt * dt * dt * dt / 12, dt * dt * dt * dt / 6, dt * dt * dt / 6 },
                    { dt * dt * dt * dt * dt / 12, dt * dt * dt * dt / 4, dt * dt * dt / 2, dt * dt / 2 },
                    { dt * dt * dt * dt / 6, dt * dt * dt / 2, dt * dt, dt },
                    { dt * dt * dt / 6, dt * dt / 2, dt, 1 }
                });
                break;
        }
        return Q * var;  
    }
    
    public static (Vector<float> mean, Matrix<float> covariance) UnscentedTransform(Matrix<float> sigmas, Vector<float> W_m, Vector<float> W_c, Matrix<float> noice_cov = null, UnscentedKalmanFilter.MeanFunction mean_fn = null,
        UnscentedKalmanFilter.ResidualFunction residual_fn = null
        )
    {
        var kmax = sigmas.RowCount;
        var n = sigmas.ColumnCount;
        
        //Debug.Log($"[{nameof(UnscentedTransform)}] kmax {kmax} n {n} input sigma is {sigmas}");
        
        Vector<float> mean = Vector<float>.Build.Dense(sigmas.ColumnCount);

        if (mean_fn != null)
        {
            mean_fn(sigmas,W_m);
        }
        else
        {
            for (int i = 0; i < sigmas.RowCount; i++)
            {
                mean += W_m[i] * sigmas.Row(i);
            }
        }

        Matrix<float> cov = Matrix<float>.Build.Dense(n, n);
        
        for (int k = 0; k < kmax; k++)
        {
            Vector<float> y;
            if (residual_fn != null)
            {
                y = residual_fn(sigmas.Row(k), mean);
            }
            else
            {
                y = sigmas.Row(k) - mean;
            }
            cov += W_c[k] * y.OuterProduct(y);
            //Debug.Log($"[{nameof(UnscentedTransform)}] step {k} result cov is {cov} ");
        }

        //Debug.Log($"[{nameof(UnscentedTransform)}] mean {mean} P is {cov} ");
        
        if (noice_cov != null)
        {
            cov = cov + noice_cov;
        }

        //Debug.Log($"[{nameof(UnscentedTransform)}] mean {mean} noised P is {cov} ");

        return (mean, cov);
    }
    
    public static List<double> GenerateTimeArray(double dt)
    {
        List<double> time = new List<double>();
        for (double t = 0; t <= 3600; t += dt)
        {
            time.Add(t);
        }
        return time;
    }
}
