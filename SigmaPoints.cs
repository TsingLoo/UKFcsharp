using System;
using MathNet.Numerics.LinearAlgebra;

public abstract class SigmaPoints
{
    protected int n;
    
    public Vector<float> W_m;
    public Vector<float> W_c;
    
    public int NumSigmas => 2 * n + 1;

    public abstract void ComputeWeights();
    public abstract Matrix<float> GenerateSigmaPoints(Vector<float> x, Matrix<float> P);
    
    public override string ToString()
    {
        return $"[{nameof(SigmaPoints)}] n: {n}, W_m: [{string.Join(", ", W_m.ToArray())}], W_c: [{string.Join(", ", W_c.ToArray())}]";
    }
}

public class JulierSigmaPoints : SigmaPoints
{
    private float kappa;
    
    public JulierSigmaPoints(int n, float kappa)
    {
        this.n = n;
        this.kappa = kappa;
        
        ComputeWeights();
    }

    public override void ComputeWeights()
    {   
        this.W_m = Vector<float>.Build.Dense(2 * n + 1, 1f / (2f * (n + kappa)));
        this.W_m[0] = kappa / (n + kappa);
        this.W_c = W_m;
    }

    public override Matrix<float> GenerateSigmaPoints(Vector<float> x, Matrix<float> P)
    {
        throw new NotImplementedException();
    }
}

public class MerweScaledSigmaPoints : SigmaPoints
{
    private float alpha;
    private float beta;
    private float kappa;
    
    public MerweScaledSigmaPoints(int n, float alpha, float beta, float kappa)
    {
        this.n = n;
        this.alpha = alpha;
        this.beta = beta;
        this.kappa = kappa;
        
        ComputeWeights();
    }

    public override void ComputeWeights()
    { 
        var lambda_ = alpha * alpha * (n + kappa) - n;
        
        var c = 0.5f / (n + lambda_);
        
        this.W_c = Vector<float>.Build.Dense(2 * n + 1, c);
        this.W_m = Vector<float>.Build.Dense(2 * n + 1, c);
        this.W_c[0] = lambda_ / (n + lambda_) + (1 - alpha * alpha + beta);
        this.W_m[0] = lambda_ / (n + lambda_);
    }
    
    public override Matrix<float> GenerateSigmaPoints(Vector<float> x, Matrix<float> P)
    {
        //Debug.Log($"[{nameof(NumSigmas)}] P raw {P}");
        
        var lambda_ = alpha * alpha * (n + kappa) - n;
        
        //Debug.Log($"Internal lambda_ is {(lambda_ + n)*P}");

        Matrix<float> U = ((lambda_ + n) * P).Transpose().Cholesky().Factor.Transpose();

        //Debug.Log($"[{nameof(GenerateSigmaPoints)}] U is {U}");
        
        Matrix<float> sigmas = Matrix<float>.Build.Dense(2 * n + 1, n);
        sigmas.SetRow(0, x);

        for (int k = 0; k < n; k++)
        {
            Vector<float> sqrtP = U.Row(k);
            //Debug.Log($"[{nameof(GenerateSigmaPoints)}] sqrtP of {k} U.Row({U.Row(k)}) is [{string.Join(", ", sqrtP.ToArray())}]");
            sigmas.SetRow(k + 1, x + sqrtP);
            sigmas.SetRow(n + k + 1, x - sqrtP);
        }

        //Debug.Log($"[{nameof(GenerateSigmaPoints)}] Generated Sigmas are {sigmas}");
        
        return sigmas;
    }
}
