% Solve a multivariate nonconvex linear constrained quadratic semidefinite programming  
% problem constrained to the unit simplex using MULTIPLE SOLVERS.
run('../../../../init.m');

n = 100;
r = 1;
m = 1e4;
M = 1e6;
global_tol = 1e-3;

% Run an instance via the command line.
print_tbls(n, r, m, M, global_tol);

%% Utility functions
function print_tbls(dimN, r, m, M, global_tol)
  seed = 777;
  dimM = 25;
  N = 1000;
  density = 0.05;
  run_experiment(N, r, M, m, dimM, dimN, density, seed, global_tol);
end
function run_experiment(N, r, M, m, dimM, dimN, density, seed, global_tol)

  [oracle, hparams] = test_fn_lin_cone_constr_04r(N, r, M, m, seed, dimM, dimN, density);
  
  % Set up the termination function. The domain is 0 <= lam_i(X) <= r.
  function proj = proj_dh(A, B)
    % Projection of `B` onto the subdifferential of `h` at `A`.
    proj = normal_eigenbox_proj(A, B, r); 
  end
  function proj = proj_NKt(~, B)
    % Projection of `B` onto the normal cone of the dual cone of `K`={0} at `A`.
    proj = zeros(size(B));
  end
  o_at_x0 = copy(oracle);
  o_at_x0.eval(hparams.x0);
  g0 = hparams.constr_fn(hparams.x0);
  rho = global_tol * (1 + hparams.norm_fn(o_at_x0.grad_f_s()));
  eta = global_tol * (1 + hparams.norm_fn(g0 - hparams.set_projector(g0)));
  alt_grad_constr_fn = @(x, p) tsr_mult(hparams.grad_constr_fn(x), p, 'dual');
  term_wrap = @(x,p) ...
    termination_check(x, p, o_at_x0, hparams.constr_fn, alt_grad_constr_fn, @proj_dh, @proj_NKt, hparams.norm_fn, rho, eta);  
  
  % Create the Model object and specify the solver.
  ncvx_qsdp = ConstrCompModel(oracle);
  
  % Set the curvatures and the starting point x0.
  ncvx_qsdp.x0 = hparams.x0;
  ncvx_qsdp.M = hparams.M;
  ncvx_qsdp.m = hparams.m;
  ncvx_qsdp.K_constr = hparams.K_constr;
  
  % Set the tolerances
  ncvx_qsdp.opt_tol = global_tol;
  ncvx_qsdp.feas_tol = global_tol;
  ncvx_qsdp.time_limit = 6000;
  
  % Add linear constraints
  ncvx_qsdp.constr_fn = hparams.constr_fn;
  ncvx_qsdp.grad_constr_fn = hparams.grad_constr_fn;
  ncvx_qsdp.set_projector = hparams.set_projector;
  
  % Use a relative termination criterion.
  ncvx_qsdp.feas_type = 'relative';
  ncvx_qsdp.opt_type = 'relative';
  
  % Create some basic hparams.
  base_hparam = struct();
  base_hparam.i_debug = true;
  base_hparam.termination_fn = term_wrap;
  base_hparam.check_all_terminations = true;
  
  % Create the IAPIAL hparams.
  ipl_hparam = base_hparam;
  ipl_hparam.acg_steptype = 'constant';
  ipla_hparam = base_hparam;
  ipla_hparam.acg_steptype = 'variable';
  
  % Run a benchmark test and print the summary.
  hparam_arr = {ipl_hparam, ipla_hparam};
  name_arr = {'IPL', 'IPL_A'};
  framework_arr = {@IAIPAL, @IAIPAL};
  solver_arr = {@ECG, @ECG};
  
  % Run the test.
  [~, summary_mdls] = run_CCM_benchmark(ncvx_qsdp, framework_arr, solver_arr, hparam_arr, name_arr);
  for s=name_arr
    disp(['Algorithm = ', s{1}]);
    disp(table(dimN, m, M, r));
    hist = summary_mdls.(s{1}).history;
    tbl = table(hist.theta1, hist.theta2, hist.inner_iter, hist.sigma, hist.c, hist.L_est, hist.norm_p_hat);
    tbl.Properties.VariableNames = ...
      {'theta1', 'theta2', 'inner_iter', 'sigma', 'penalty_param', 'L_est', 'norm_p_hat'};
    disp(tbl);
  end
  
end