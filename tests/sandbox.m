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
foo = ConstrCompModel(oracle);
foo.solver = @ECG;

% Add penalty parameters
foo.K_constr = 0.1;
foo.framework = @penalty;

% Set the curvatures and the starting point x0.
foo.M = hparams.M;
foo.m = hparams.m;
foo.x0 = hparams.x0;

% Solve the problem.
foo.optimize;