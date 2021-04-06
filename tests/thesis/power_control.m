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
seed = 777;
sigma = sqrt(0.5);
rho_x = 1e-1;
rho_y = 1e-1;
time_limit = 4000;

% -------------------------------------------------------------------------
%% Table 1
% -------------------------------------------------------------------------
disp('========')
disp('TABLE 1');
disp('========')
N_vec = [5, 10, 25, 50];
K_vec = [5, 10, 25, 50];
% Loop over the upper curvature M.
for i = 1:length(N_vec)
  dim_N = N_vec(i);
  dim_K = K_vec(i);
  % Use a problem instance generator to create the oracle and
  % hyperparameters.
  [oracle_factory, hparams] = ...
    test_fn_power_control_01(dim_K, dim_N, sigma, seed);
  hparams.rho_y = rho_y;
  [xi, M_v1] = compute_smoothed_parameters(hparams, 'AIPP-S');
  [~,  M_v2] = compute_smoothed_parameters(hparams, 'PGSF');
  oracle = oracle_factory(xi);

  % Create the Model object and specify the solver.
  ncvx_power_control = CompModel(oracle);

  % Set the curvatures and the starting point x0.
  ncvx_power_control.M = M_v1;
  ncvx_power_control.m = hparams.m;
  ncvx_power_control.x0 = hparams.x0;

  % Set up the termination criterion.
  ncvx_power_control.opt_type = 'relative';
  ncvx_power_control.opt_tol = rho_x;
  ncvx_power_control.time_limit = time_limit;
  
  % Split the model for PGSF.
  pgsf_model = copy(ncvx_power_control);
  pgsf_model.M = M_v2;

  % Run a benchmark test and print the summary.
  oracle_arr = {pgsf_model, ncvx_power_control, ncvx_power_control};
  solver_arr = {@ECG, @AG, @AIPP};
  hparam_arr = {base_hparam, base_hparam, aipp_hparam};
  name_arr = {'PGSF', 'AG', 'R_AIPP'};
  [summary_tables, comp_models] = ...
    run_CM_benchmark(ncvx_power_control, solver_arr, hparam_arr, name_arr);
  
  % Set up the final table.
  sub_table = summary_tables.all;
  sub_table.dim_N = dim_N;
  sub_table.dim_K = dim_K;
  disp(sub_table);
  if (i == 1)
    final_table = sub_table;
  else
    final_table = [final_table; sub_table];
  end
end

% Display final table for logging.
disp(final_table);