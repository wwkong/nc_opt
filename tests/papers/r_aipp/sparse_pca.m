% Solve a sparse PCA problem using MULTIPLE SOLVERS.

% Set up paths.
run('../../../init.m');

% -------------------------------------------------------------------------
%% Global Variables
% -------------------------------------------------------------------------

% Create basic hparams.
base_hparam = struct();
aipp_hparam = base_hparam;
aipp_hparam.tau = 100000; % SET A SPECIAL VALUE OF TAU
aipp_c_hparam = aipp_hparam;
aipp_c_hparam.aipp_type = 'aipp_c';
aipp_v1_hparam = aipp_hparam;
aipp_v1_hparam.aipp_type = 'aipp_v1';
aipp_v2_hparam = aipp_hparam;
aipp_v2_hparam.aipp_type = 'aipp_v2';

% Create global hyperparams
b = 0.1;
nu = 100;
p = 100;
n = 100;
seed = 7777;
global_opt_tol = 1e-6;
global_feas_tol = 1e-3;
time_limit = 4000;

% -------------------------------------------------------------------------
%% Table
% -------------------------------------------------------------------------
% Loop over the upper curvature M.
sk_vec = ...
  [5,  1;
   10, 1;
   15, 1];
[nrows, ncols] = size(sk_vec);
for i = 1:nrows
  % Use a problem instance generator to create the oracle and
  % hyperparameters.
  s = sk_vec(i, 1);
  k = sk_vec(i, 2);
  [oracle, hparams] = ...
    test_fn_spca_01(b, nu, p, n, s, k, seed);

  % Create the Model object and specify the solver.
  spca = ConstrCompModel(oracle);

  % Set the curvatures, the starting point x0, and special functions.
  spca.M = hparams.M;
  spca.m = hparams.m;
  spca.x0 = hparams.x0;
  spca.K_constr = hparams.K_constr;
  
  % Add linear constraints.
  spca.constr_fn = @(x) hparams.constr_fn(x);
  spca.grad_constr_fn = hparams.grad_constr_fn;

  % Set up the termination criterion.
  spca.opt_type = 'relative';
  spca.feas_type = 'absolute';
  spca.opt_tol = global_opt_tol;
  spca.feas_tol = global_feas_tol;
  spca.time_limit = time_limit;

  % Run a benchmark test and print the summary.
  solver_arr = ...
    {@UPFAG, @NC_FISTA, @AG, @AIPP, @AIPP, @AIPP};
  hparam_arr = ...
    {base_hparam, base_hparam, base_hparam, aipp_c_hparam, ...
     aipp_v1_hparam, aipp_v2_hparam};
  name_arr = ...
    {'UPFAG', 'NC_FISTA', 'AG', 'AIPP_c', 'AIPP_v1', 'AIPP_v2'};
  framework_arr = ...
    {@penalty, @penalty, @penalty, @penalty, @penalty, @penalty}; 
  [summary_tables, comp_models] = ...
    run_CCM_benchmark(...
      spca, framework_arr, solver_arr, hparam_arr, name_arr);
  
   % Set up the final table.
  sub_table = summary_tables.all;
  sub_table.s = s;
  disp(sub_table);
  if (i == 1)
    final_table = sub_table;
  else
    final_table = [final_table; sub_table];
  end
end

% Display final table for logging.
disp(final_table);
