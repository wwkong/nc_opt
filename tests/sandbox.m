% Test generator
N = 1000;
M = 100;
m = 5;
seed = 777;
dimM = 5;
dimN = 20;
[oracle, hparams] = test_fn_unconstr_01(N, M, m, seed, dimM, dimN);
oracle.eval(rand(dimN, 1));

% Create the Model object and specify the solver.
ncvx_qp = CompModel(oracle);
ncvx_qp.solver = @ADAP_FISTA;

% Set the curvatures and the starting point x0.
ncvx_qp.L = max([abs(hparams.M), abs(hparams.m)]);
ncvx_qp.M = hparams.M;
hparams.m = hparams.m;
ncvx_qp.x0 = hparams.z0;

% Solve the problem.
ncvx_qp.optimize;