using MathNet.Numerics.LinearAlgebra;

public class UnscentedKalmanFilter
{
    public Vector<float> x;
    public Matrix<float> P;

    private Vector<float> x_prior;
    private Matrix<float> P_prior;
    
    private Vector<float> x_post;
    private Matrix<float> P_post;

    private Matrix<float> Q;
    private Matrix<float> R;

    private int dim_x;
    private int dim_z;
    
    private SigmaPoints points_fn;

    Matrix<float> sigmas_f;
    Matrix<float> sigmas_h;

    private float dt;
    private int num_sigmas;
    
    Matrix<float> K;
    Vector<float> y;
    Vector<float> z;
    Matrix<float> S;
    Matrix<float> SI;
    
    // Delegates for the state and measurement functions
    public delegate Vector<float> StateTransitionModel(Vector<float> x, float dt, Vector<float> u = null, object args = null);
    public delegate Vector<float> MeasurementModel(Vector<float> x, object args = null);

    // Delegates for mean functions
    public delegate Vector<float> MeanFunction(Matrix<float> sigmas, Vector<float> weight);
    public delegate Vector<float> ResidualFunction(Vector<float> a, Vector<float> b);

    private StateTransitionModel fx;
    private MeasurementModel hx;
    private MeanFunction x_mean;
    private MeanFunction z_mean;

    private ResidualFunction residual_x;
    private ResidualFunction residual_z;

    public UnscentedKalmanFilter(int dim_x, int dim_z, float dt,
        StateTransitionModel fx, MeasurementModel hx, SigmaPoints points,
        MeanFunction xMeanFn = null, MeanFunction zMeanFn= null, ResidualFunction residualX = null, ResidualFunction residualZ = null)
    {
        this.dim_x = dim_x;
        this.dim_z = dim_z;
        this.dt = dt;

        this.x = Vector<float>.Build.Dense(dim_x);
        this.P = Matrix<float>.Build.DenseIdentity(dim_x);

        this.x_prior = Vector<float>.Build.Dense(dim_x);
        this.P_prior = Matrix<float>.Build.DenseIdentity(dim_x);
        this.x_post = Vector<float>.Build.Dense(dim_x);
        this.P_post = Matrix<float>.Build.DenseIdentity(dim_x);

        this.Q = KFUtils.Q_DiscreteWhiteNoise(2, 1.0f, 0.01f);
        this.R = Matrix<float>.Build.DenseIdentity(dim_z);

        this.points_fn = points;
        this.num_sigmas = points_fn.NumSigmas;

        this.fx = fx;
        this.hx = hx;
        this.x_mean = xMeanFn;
        this.z_mean = zMeanFn;

        sigmas_f = Matrix<float>.Build.Dense(num_sigmas, dim_x);
        sigmas_h = Matrix<float>.Build.Dense(num_sigmas, dim_z);

        if (residualX == null)
        {
            this.residual_x = (a, b) => a - b;
        }

        if (residualZ == null)
        {
            this.residual_z = (a, b) => a - b;
        }

        this.K = Matrix<float>.Build.Dense(dim_x, dim_x);
        this.y = Vector<float>.Build.Dense(dim_z);
        this.z = Vector<float>.Build.Dense(dim_z);
        this.S = Matrix<float>.Build.Dense(dim_z, dim_z);
        this.SI = Matrix<float>.Build.Dense(dim_z, dim_z);
    }
    
    public void Set_x(Vector<float> x)
    {
        if(x.Count != dim_x)
            //Debug.LogError($"[{nameof(UnscentedKalmanFilter)}] x must have the same dimension as dim_x");
        this.x = x;
    }
    
    public void Set_P(Matrix<float> P)
    {
        if(P.RowCount != dim_x || P.ColumnCount != dim_x)
            //Debug.LogError($"[{nameof(UnscentedKalmanFilter)}] P must have the same dimension as dim_x");
        this.P = P;
    }
    
    public void Set_R(Matrix<float> R)
    {
        if(R.RowCount != dim_z || R.ColumnCount != dim_z)
            //Debug.LogError($"[{nameof(UnscentedKalmanFilter)}] R must have the same dimension as dim_z");
        this.R = R;
    }
    
    public void Set_Q(Matrix<float> Q)
    {
        if(Q.RowCount != dim_x || Q.ColumnCount != dim_x)
            //Debug.LogError($"[{nameof(UnscentedKalmanFilter)}] Q must have the same dimension as dim_x");
        this.Q = Q;
    }

    public void Predict()
    {
        ComputeProcessSigmas(dt,fx);

        var predicted = KFUtils.UnscentedTransform(sigmas_f, points_fn.W_m, points_fn.W_c, Q);
        x = predicted.mean;
        P = predicted.covariance;

        //Debug.Log($"[{nameof(UnscentedKalmanFilter)}] Predicted x is {x} P is {P}");
        
        x.CopyTo(x_prior);
        P.CopyTo(P_prior);
    }

    public void Update(Vector<float> z, Matrix<float> R = null, MeasurementModel hx = null, object args = null)
    {
        if (R == null)
            R = this.R;
        
        //Debug.Log($"[{nameof(UnscentedKalmanFilter)}] R is {this.R}");
        
        if (hx == null)
            hx = this.hx;
        
        for(int i = 0; i<sigmas_f.RowCount; i ++ )
        {
            sigmas_h.SetRow(i, hx(sigmas_f.Row(i), args));
        }

        var (zp ,S) = KFUtils.UnscentedTransform(sigmas_h, points_fn.W_m, points_fn.W_c, R, z_mean);
        this.S = S;
        this.SI = S.Inverse();
        
        //Debug.Log($"[{nameof(UnscentedKalmanFilter)}] S is {this.S} SI is {this.SI}");
        
        var Pxz = CrossVariance(this.x,zp, sigmas_f, sigmas_h);

        this.K = Pxz * this.SI;
        this.y = residual_z(z, zp);
        
        //Debug.Log($"[{nameof(UnscentedKalmanFilter)}] Updated K is {K} y is {this.y}");
        
        this.x = this.x + this.K * this.y;
        
        //Debug.Log($"[{nameof(UnscentedKalmanFilter)}] Raw P is {P}");
        
        this.P = this.P - this.K * (this.S * this.K.Transpose());
        
        //Debug.Log($"[{nameof(UnscentedKalmanFilter)}] P internal sub is {(this.S * this.K.Transpose())}");
        //Debug.Log($"[{nameof(UnscentedKalmanFilter)}] P out sub is { this.K * (this.S * this.K.Transpose())}");

        
        // Debug.Log($"[{nameof(UnscentedKalmanFilter)}] Updated Raw x is {x} an P is {P}");
        
        this.z = z;
        this.x.CopyTo(x_post);
        this.P.CopyTo(P_post);
        
        //Debug.Log($"[{nameof(UnscentedKalmanFilter)}] Updated x is {x} an P is {P}");
        //Debug.LogWarning($"[{nameof(UnscentedKalmanFilter)}]");
    }

    void ComputeProcessSigmas(float dt, StateTransitionModel fx = null, object args = null)
    { 
        var sigmas = points_fn.GenerateSigmaPoints(x, P);
        
        //Debug.Log($"[{nameof(ComputeProcessSigmas)}] Internal sigmas is {sigmas}");
        
        for (int i = 0; i < sigmas.RowCount; i++)
        {
            Vector<float> sigmaPoint = sigmas.Row(i);
            this.sigmas_f.SetRow(i, fx(sigmaPoint, dt));
            
            //Debug.Log($"[{nameof(ComputeProcessSigmas)}] sigmas_f's {i}Row is {sigmas_f.Row(i)}");
        }

        //Debug.Log($"[{nameof(ComputeProcessSigmas)}] sigmas_f is {sigmas_f}");
    }
    
    public Matrix<float> CrossVariance(Vector<float> x, Vector<float> z, Matrix<float> sigmas_f, Matrix<float> sigmas_h)
    {
        int stateDim = sigmas_f.ColumnCount;
        int measurementDim = sigmas_h.ColumnCount;
        Matrix<float> Pxz = Matrix<float>.Build.Dense(stateDim, measurementDim);
        int N = sigmas_f.RowCount;

        for (int i = 0; i < N; i++)
        {
            //Debug.Log($"[{nameof(CrossVariance)}] Pxz is {Pxz}");
            //Debug.Log($"[{nameof(CrossVariance)}] Pxz sigmas_f is {sigmas_f.Row(i)},x is {x}");
            //Debug.Log($"[{nameof(CrossVariance)}] Pxz sigmas_h is {sigmas_h.Row(i)},z is {z}");
            Vector<float> dx = residual_x(sigmas_f.Row(i),x);
            Vector<float> dz = residual_z(sigmas_h.Row(i),z);
            Pxz += points_fn.W_c[i] * dx.OuterProduct(dz);
        }

        //Debug.Log($"[{nameof(CrossVariance)}] Pxz is {Pxz}");
        
        return Pxz;
    }
}
