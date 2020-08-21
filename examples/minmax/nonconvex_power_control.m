% Solve a nonconvex power control problem.

% Use a problem instance generator to create the oracle factory and
% hyperparameters.
sigma = sqrt(0.5);
dim_K = 5;
dim_N = 5;
[oracle_factory, hparams] = ...
  test_fn_power_control_01(dim_K, dim_N, sigma, seed);

% Set up the oracle and special tolerances.
hparams.rho_x = 1e-1;
hparams.rho_y = 1e-1;
[xi, M_smoothed] = compute_smoothed_parameters(hparams, 'AIPP-S');
oracle = oracle_factory(xi);

% Create the Model object and specify the solver and tolerance.
ncvx_pc = CompModel(oracle);
ncvx_pc.solver = @AIPP;
ncvx_pc.opt_tol = hparams.rho_x;

% Set the curvatures and the starting point x0.
ncvx_pc.M = M_smoothed;
ncvx_pc.m = hparams.m;
ncvx_pc.x0 = hparams.x0;

% Use a relative termination criterion.
ncvx_pc.opt_type = 'relative';

% Solve the problem.
ncvx_pc.optimize;