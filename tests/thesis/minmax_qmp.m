% Solve a multivariate minmax nonconvex quadratic programming problem 
% constrained to the unit simplex using MULTIPLE SOLVERS.

% The function of interest is
%
%   f(x) :=  max_i 
%     {
%       -xi / 2 * ||D_i * B_i * x|| ^ 2 + tau / 2 * ||A_i * x - b_i|| ^ 2
%     }
%
% with curvature pairs {(m_i, M_i)}. 

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
N = 1000;
seed = 777;
dimM = 10;
dimN = 200;
density = 0.05;
k = 5;
rho_x = 1e-2;
rho_y = 1e-1;
time_limit = 4000;

% -------------------------------------------------------------------------
%% Table 1
% -------------------------------------------------------------------------
disp('========')
disp('TABLE 1');
disp('========')
M_vec = [1e1, 1e2, 1e3, 1e3];
% Loop over the upper curvature M.
for i = 1:length(M_vec)
  M = M_vec(i);
  m = 1e1;
  % Use a problem instance generator to create the oracle and
  % hyperparameters.
  [oracle_factory, hparams] = ...
    test_fn_mm_unconstr_01(k, N, M, m, seed, dimM, dimN, density);
  hparams.rho_y = rho_y;
  [xi, M_v1] = compute_smoothed_parameters(hparams, 'AIPP-S');
  [~,  M_v2] = compute_smoothed_parameters(hparams, 'PGSF');
  oracle = oracle_factory(xi);

  % Create the Model object and specify the solver.
  mm_qmp = CompModel(oracle);

  % Set the curvatures and the starting point x0.
  mm_qmp.M = M_v1;
  mm_qmp.m = hparams.m;
  mm_qmp.x0 = hparams.x0;

  % Set up the termination criterion.
  mm_qmp.opt_type = 'relative';
  mm_qmp.opt_tol = rho_x;
  mm_qmp.time_limit = time_limit;
  
  % Split the model for PGSF.
  pgsf_model = copy(mm_qmp);
  pgsf_model.M = M_v2;

  % Run a benchmark test and print the summary.
  oracle_arr = {pgsf_model, mm_qmp, mm_qmp};
  solver_arr = {@ECG, @AG, @AIPP};
  hparam_arr = {base_hparam, base_hparam, aipp_hparam};
  name_arr = {'PGSF', 'AG', 'R_AIPP'};
  [summary_tables, comp_models] = ...
    run_CM_benchmark(mm_qmp, solver_arr, hparam_arr, name_arr);
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