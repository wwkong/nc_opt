%% Solve a blockwise spectral matrix completion instance.

% Set basic hyperparameters for generating the matrix array.
seed = 777;
dim_m = 25;
dim_n = 50;
density = 0.25;
n_blocks = 5 * 2;
min_rating = 0;
max_rating = 10;
distribution_type = 'truncated_normal';

% Create the matrix array.
rng(seed)
matrix_arr = cell(n_blocks, 1);
if strcmp(distribution_type, 'binomial')
  for i=1:n_blocks
    p  = rand();
    n = max_rating - min_rating;
    mask = (sprand(dim_m, dim_n, density) ~= 0);
    ratings =  binornd(n, p, dim_m, dim_n) + min_rating;
    matrix_arr{i} = ratings .* mask;
  end
elseif strcmp(distribution_type, 'truncated_normal')
  for i=1:n_blocks
    p = rand();
    n = max_rating - min_rating;
    normal_pd = makedist('Normal', 'mu', n * p, 'sigma', n * p * (1 - p));
    normal_tpd = truncate(normal_pd, min_rating, max_rating);
    mask = (sprand(dim_m, dim_n, density) ~= 0);
    ratings = random(normal_tpd, dim_m, dim_n) + min_rating;
    matrix_arr{i} = ratings .* mask;
  end
else
  error('Unknown distribution type!');
end

%% Generate the test instance
alpha = 10;
beta = 20;
mu = 2;
theta = 1;
[spectral_oracle, hparams] = ...
  test_fn_spectral_bmc_01(matrix_arr, alpha, beta, theta, mu, seed);

%% Create the Model object and specify the solver.
ncvx_smc = CompModel(spectral_oracle);
ncvx_smc.solver = @IA_ICG;
ncvx_smc.time_limit = 5;
ncvx_smc.opt_type = 'relative';
ncvx_smc.opt_tol = 1e-6;

% Set the curvatures and the starting point x0.
ncvx_smc.solver_hparams = hparams;
ncvx_smc.M = hparams.M;
ncvx_smc.m = hparams.m;
ncvx_smc.x0 = hparams.x0;

% Solve the problem.
ncvx_smc.optimize;