% Solve a multivariate nonconvex quadratically constrained quadratic programming  
% problem constrained to a box using MULTIPLE SOLVERS.
run('../../../../init.m');
format long

% Comment out later.
dimN = 500;
r = 1; 
m = 1e1; 
M = 1e5;

% Run an instance via the command line.
print_tbls(dimN, r, m, M);

%% Utility functions
function print_tbls(dimN, r, m, M) 

  % Initialize
  seed = 77777;
  dimM = 10;
  global_tol = 1e-5;
  disp(table(dimN, r, m, M));
  run_experiment(M, m, dimM, dimN, -r, r, seed, global_tol);
  
end
function run_experiment(M, m, dimM, dimN, x_l, x_u, seed, global_tol)

  [oracle, hparams] = test_fn_quad_box_constr_02(M, m, seed, dimM, dimN, x_l, x_u);
  
  % Set up the termination function.
  function proj = proj_dh(a, b)
    I1 = (abs(a - x_l) < 1e-12); 
    I2 = (abs(a - x_u) <= 1e-12);
    I3 = (abs(a - x_l) > 1e-12 & abs(a - x_u) > 1e-12);
    proj = b;
    proj(I1) = min(0, b(I1));
    proj(I2) = max(0, b(I2));
    proj(I3) = 0;
  end
  function proj = proj_NKt(~, b)
    proj = min(0, b);
  end
  o_at_x0 = copy(oracle);
  o_at_x0.eval(hparams.x0);
  g0 = hparams.constr_fn(hparams.x0);
  rho = global_tol * (1 + hparams.norm_fn(o_at_x0.grad_f_s()));
  eta = global_tol * (1 + hparams.norm_fn(g0 - hparams.set_projector(g0)));
  term_wrap = @(x,p) ...
    termination_check(x, p, o_at_x0, hparams.constr_fn, hparams.grad_constr_fn, @proj_dh, @proj_NKt, hparams.norm_fn, rho, eta);

  % Create the Model object and specify the solver.
  ncvx_qc_qp = ConstrCompModel(oracle);
  
  % Set the curvatures and the starting point x0.
  ncvx_qc_qp.x0 = hparams.x0;
  ncvx_qc_qp.M = hparams.M;
  ncvx_qc_qp.m = hparams.m;
  ncvx_qc_qp.K_constr = hparams.K_constr;
  ncvx_qc_qp.L_constr = hparams.L_constr;
  
  % Set the tolerances
  ncvx_qc_qp.opt_tol = global_tol;
  ncvx_qc_qp.feas_tol = global_tol;
  ncvx_qc_qp.time_limit = 2000;
  ncvx_qc_qp.iter_limit = 1000000;
  
  % Add linear constraints
  ncvx_qc_qp.constr_fn = hparams.constr_fn;
  ncvx_qc_qp.grad_constr_fn = hparams.grad_constr_fn;
  ncvx_qc_qp.set_projector = hparams.set_projector;
  ncvx_qc_qp.dual_cone_projector = hparams.dual_cone_projector;
  
  % Use a relative termination criterion.
  ncvx_qc_qp.feas_type = 'relative';
  ncvx_qc_qp.opt_type = 'relative';
  
  % Create some basic hparams.
  base_hparam = struct();
  base_hparam.i_debug = true;
  base_hparam.termination_fn = term_wrap;
  base_hparam.check_all_terminations = true;
  
  % Create the IAPIAL hparams.
  ipl_hparam = base_hparam;
  ipl_hparam.sigma_min = sqrt(0.3);
  ipl_hparam.acg_steptype = 'constant';
  ipla_hparam = ipl_hparam;
  ipla_hparam.L_start = (hparams.M / (2 * hparams.m) + 1) / 1000;
  ipla_hparam.acg_steptype = 'variable';
  
  % Run a benchmark test and print the summary.
  hparam_arr = {ipla_hparam, ipl_hparam};
  name_arr = {'IPL_A', 'IPL'};
  framework_arr = {@IAIPAL, @IAIPAL};
  solver_arr = {@ECG, @ECG};
  
  % Run the test.
  [summary_tbls, summary_mdls] = run_CCM_benchmark(ncvx_qc_qp, framework_arr, solver_arr, hparam_arr, name_arr);
  for s=name_arr
    disp(['Algorithm = ', s{1}]);
    disp(table(dimN, m, M, x_l, x_u));
    hist = summary_mdls.(s{1}).history;
    tbl = table(hist.inner_iter, hist.c, hist.L_est, hist.norm_v, hist.norm_w);
    tbl.Properties.VariableNames = {'inner_iter', 'penalty_param', 'L_est', 'opt_resid', 'feas_resid'};
    disp(tbl);
  end
  disp(summary_tbls.all);

end

