% Test generator
N = 1000;
M = 100;
m = 1;
seed = 777;
dimM = 5;
dimN = 10;
[oracle, hparams] = test_fn_unconstr_01(N, M, m, seed, dimM, dimN);

% Create the Model object and specify the solver.
foo = ConstrCompModel(oracle);
foo.solver = @AC_ACG;
foo.feas_type = 'relative';

% Add linear constraints
A = rand(dimM * 2, dimN);
b = A * ones(dimN, 1) / dimN;
foo.constr_fn = @(x) A * x;
foo.grad_constr_fn = @(x) A';
foo.set_projector = @(x) b;

% Add penalty framework
foo.K_constr = norm(A, 2);
foo.opt_tol = 1e-2;
foo.feas_tol = 1e-2;
foo.framework = @penalty;

% Set the curvatures and the starting point x0.
foo.M = hparams.M;
foo.m = hparams.m;
foo.x0 = hparams.x0;

% Solve the problem.
foo.optimize;