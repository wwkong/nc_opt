% Solve a multivariate nonconvex quadratic programming problem 
% constrained to the unit simplex using MULTIPLE SOLVERS.

% The function of interest is
%
%  f(x) :=  -xi / 2 * ||D * B * x|| ^ 2 + tau / 2 * ||A * x - b|| ^ 2
%
% with curvature pair (m, M). 

% Set up paths.
run('../../../init.m');

% -------------------------------------------------------------------------
%% Global Variables
% -------------------------------------------------------------------------

% Create basic hparams.
base_hparam = struct();
aipp_hparam = base_hparam;
aipp_hparam.tau = 10000; % SET A SPECIAL VALUE OF TAU
aipp_c_hparam = aipp_hparam;
aipp_c_hparam.aipp_type = 'aipp_c';
aipp_v1_hparam = aipp_hparam;
aipp_v1_hparam.aipp_type = 'aipp_v1';
aipp_v2_hparam = aipp_hparam;
aipp_v2_hparam.aipp_type = 'aipp_v2';

% Create global hyperparams
N = 1000;
seed = 777;
dimM = 50;
dimN = 200;
density = 0.025;
global_tol = 1e-7;
time_limit = 4000;

% -------------------------------------------------------------------------
%% Tables
% -------------------------------------------------------------------------
% Loop over the curvature pair (m, M).
mM_vec = ...
  [1e0, 1e2;
   1e0, 1e4;
   1e0, 1e6;
   1e2, 1e2;
   1e4, 1e4;
   1e6, 1e6];
[nrows, ncols] = size(mM_vec);
for i = 1:nrows
  % Use a problem instance generator to create the oracle and
  % hyperparameters.
  m = mM_vec(i, 1);
  M = mM_vec(i, 2);
  [oracle, hparams] = ...
    test_fn_unconstr_02(N, M, m, seed, dimM, dimN, density);

  % Create the Model object and specify the solver.
  ncvx_qp = CompModel(oracle);

  % Set the curvatures and the starting point x0.
  ncvx_qp.M = hparams.M;
  ncvx_qp.m = hparams.m;
  ncvx_qp.x0 = hparams.x0;

  % Set up the termination criterion.
  ncvx_qp.opt_type = 'relative';
  ncvx_qp.opt_tol = global_tol;
  ncvx_qp.time_limit = time_limit;

  % Run a benchmark test and print the summary.
  solver_arr = ...
    {@UPFAG, @NC_FISTA, @AG, @AIPP, @AIPP, @AIPP};
  hparam_arr = ...
    {base_hparam, base_hparam, base_hparam, aipp_c_hparam, ...
     aipp_v1_hparam, aipp_v2_hparam};
  name_arr = {'UPFAG', 'NC_FISTA', 'AG', 'AIPP_c', ...
    'AIPP_v1', 'AIPP_v2'};
  [summary_tables, comp_models] = ...
    run_CM_benchmark(ncvx_qp, solver_arr, hparam_arr, name_arr);
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
