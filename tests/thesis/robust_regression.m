% Solve a nonconvex power control problem using MULTIPLE SOLVERS.

% Set up paths.
run('../../init.m');

% -------------------------------------------------------------------------
%% Global Variables
% -------------------------------------------------------------------------

% Create basic hparams.
base_hparam = struct();
aipp_hparam = base_hparam;
aipp_hparam.aipp_type = 'aipp_v2';

% Create global hyperparams
alpha = 10;
rho_x = 1e-5;
rho_y = 1e-3;
time_limit = 4000;

% -------------------------------------------------------------------------
%% Table 1
% -------------------------------------------------------------------------
disp('========')
disp('TABLE 1');
disp('========')
data_name_arr = ...
  {'../../data/heart_scale.txt', ...
   '../../data/diabetes_scale.txt', ...
   '../../data/ionosphere_scale.txt', ...
   '../../data/sonar_scale.txt', ...
   '../../data/breast-cancer_scale.txt'};
% Loop over the upper curvature M.
for i = 1:length(data_name_arr)
  data_name = data_name_arr{i};
  % Use a problem instance generator to create the oracle and
  % hyperparameters.
  [oracle_factory, hparams] = ...
    test_fn_robust_regression_01(data_name, alpha);
  hparams.rho_y = rho_y;
  [xi, M_v1] = compute_smoothed_parameters(hparams, 'AIPP-S');
  [~,  M_v2] = compute_smoothed_parameters(hparams, 'PGSF');
  oracle = oracle_factory(xi);

  % Create the Model object and specify the solver.
  ncvx_rr = CompModel(oracle);

  % Set the curvatures and the starting point x0.
  ncvx_rr.M = M_v1;
  ncvx_rr.m = hparams.m;
  ncvx_rr.x0 = hparams.x0;

  % Set up the termination criterion.
  ncvx_rr.opt_type = 'relative';
  ncvx_rr.opt_tol = rho_x;
  ncvx_rr.time_limit = time_limit;
  
  % Split the model for PGSF.
  pgsf_model = copy(ncvx_rr);
  pgsf_model.M = M_v2;

  % Run a benchmark test and print the summary.
  oracle_arr = {pgsf_model, ncvx_rr, ncvx_rr};
  solver_arr = {@ECG, @AG, @AIPP};
  hparam_arr = {base_hparam, base_hparam, aipp_hparam};
  name_arr = {'PGSF', 'AG', 'R_AIPP'};
  [summary_tables, comp_models] = ...
    run_CM_benchmark(ncvx_rr, solver_arr, hparam_arr, name_arr);
  disp(summary_tables.all);
  
  % Set up the final table.
  if (i == 1)
    final_table = summary_tables.all;
  else
    final_table = [final_table; summary_tables.all];
  end
end

% Display final table for logging.
disp(final_table);