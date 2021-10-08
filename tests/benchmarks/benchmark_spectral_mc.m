% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

% Spectral matrix completion instance

% Set up paths.
run('../../init.m');

% % Instance parameters (will be set via the command line).
% data_name = 'movielens_100k_610u_9724m';
% theta = 0.1;
fprintf(['Running test instance with: \n', 'theta = ', num2str(theta), '\ndata_name = ', data_name, '\n']);

% Test generator.
alpha = 10;
beta = 20;
mu = 2;
seed = 777;
[spectral_oracle, hparams] = test_fn_spectral_mc_01(data_name, alpha, beta, theta, mu, seed);

% Create the Model object and specify the solver.
ncvx_smc = CompModel(spectral_oracle);
ncvx_smc.time_limit = 1000;
ncvx_smc.opt_type = 'relative';
ncvx_smc.opt_tol = 1e-6;

% Other global hyperparameters.
hparams.steptype = 'adaptive';
hparams.i_logging = true;

% Set the curvatures and the starting point x0.
ncvx_smc.solver_hparams = hparams;
ncvx_smc.M = hparams.M;
ncvx_smc.m = hparams.m;
ncvx_smc.x0 = hparams.x0;

% Run a benchmark test and print the summary.
solver_arr = {@ECG, @AIPP, @AG, @UPFAG, @NC_FISTA, @IA_ICG, @DA_ICG};
[summary_tables, comp_models] = run_CM_benchmark(ncvx_smc, solver_arr, [], []);
disp(summary_tables.all);

%% Save
fname = [data_name, '_theta', num2str(theta, '%1.s'), '_history.mat'];
comp_history = get_mdl_history(comp_models);
save(fname, 'comp_history');

% Helper Functions
function out_history = get_mdl_history(models)
  fnames = fieldnames(models);
  for i=1:length(fnames)
    out_history.(fnames{i}) = models.(fnames{i}).history;
  end    
end
