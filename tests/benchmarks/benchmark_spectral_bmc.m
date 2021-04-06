% Spectral matrix completion instance

% Set up paths.
run('../../init.m');

% % Instance parameters (will be set via the command line).
% distribution_type = 'truncated_normal';
% theta = 0.1;
fprintf(...
  ['Running test instance with: \n', ...
   'theta = ', num2str(theta), '\n' ...
   'distribution_type = ', distribution_type, '\n']);

% Parameters for the matrix array.
seed = 777;
dim_m = 50;
dim_n = 100;
density = 0.25;
n_blocks = 5 * 5;
min_rating = 0;
max_rating = 10;

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

% Test generator.
alpha = 10;
beta = 20;
mu = 2;
[spectral_oracle, hparams] = ...
  test_fn_spectral_bmc_01(matrix_arr, alpha, beta, theta, mu, seed);

% Create the Model object and specify the solver.
ncvx_sbmc = CompModel(spectral_oracle);
ncvx_sbmc.time_limit = 1000;
ncvx_sbmc.opt_type = 'relative';
ncvx_sbmc.opt_tol = 1e-6;

% Other global hyperparameters.
hparams.steptype = 'adaptive';
hparams.i_logging = true;

% Set the curvatures and the starting point x0.
ncvx_sbmc.solver_hparams = hparams;
ncvx_sbmc.M = hparams.M;
ncvx_sbmc.m = hparams.m;
ncvx_sbmc.x0 = hparams.x0;

% Run a benchmark test and print the summary.
solver_arr = {@ECG, @AIPP, @AG, @UPFAG, @NC_FISTA, @IA_ICG, @DA_ICG};
[summary_tables, comp_models] = ...
  run_CM_benchmark(ncvx_sbmc, solver_arr, [], []);
disp(summary_tables.all);

%% Save
fname = ...
  [distribution_type, '_theta', num2str(theta, '%1.s'), '_history.mat'];
comp_history = get_mdl_history(comp_models);
save(fname, 'comp_history');

% Helper Functions
function out_history = get_mdl_history(models)
  fnames = fieldnames(models);
  for i=1:length(fnames)
    out_history.(fnames{i}) = models.(fnames{i}).history;
  end    
end