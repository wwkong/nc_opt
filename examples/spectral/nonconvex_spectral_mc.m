% blockwise spectral matrix completion instance

% Test generator
data_name = 'jester_24938u_100j';
alpha = 10;
beta = 20;
mu = 2;
theta = 1;
seed = 777;
[spectral_oracle, hparams] = ...
  test_fn_spectral_mc_01(data_name, alpha, beta, theta, mu, seed);

% Create the Model object and specify the solver.
ncvx_smc = CompModel(spectral_oracle);
ncvx_smc.solver = @DA_ICG;
ncvx_smc.time_limit = 10;
ncvx_smc.opt_type = 'relative';
ncvx_smc.opt_tol = 1e-6;

% Set the curvatures and the starting point x0.
ncvx_smc.solver_hparams = hparams;
ncvx_smc.M = hparams.M;
ncvx_smc.m = hparams.m;
ncvx_smc.x0 = hparams.x0;

% Solve the problem.
ncvx_smc.optimize;