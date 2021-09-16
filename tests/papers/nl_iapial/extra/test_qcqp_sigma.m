% Solve a multivariate nonconvex quadratically constrained quadratic programming  
% problem constrained to a box using MULTIPLE SOLVERS.
run('../../../../init.m');

% Run an instance via the command line.
print_tbls(500, 0.01);
print_tbls(500, 0.10);
print_tbls(500, 0.20);
print_tbls(500, 0.30);
print_tbls(500, 0.40);
print_tbls(500, 0.50);
print_tbls(500, 0.60);
print_tbls(500, 1/sqrt(2));

%% Utility functions
function print_tbls(dimN, sigma_min) 

  % Initialize
  seed = 777;
  dimM = 5;
  global_tol = 1e-5;

  % Variable M.
  m = 1e4;
  M = 1e6;
  r = 1;
  o_tbl = run_experiment(M, m, dimM, dimN, -r, r, sigma_min, seed, global_tol);
  disp(['Tables for dimN = ', num2str(dimN)]);
  disp(o_tbl);
  
end
function o_tbl = run_experiment(M, m, dimM, dimN, x_l, x_u, sigma_min, seed, global_tol)

  [oracle, hparams] = ...
    test_fn_quad_box_constr_02(M, m, seed, dimM, dimN, x_l, x_u);
  
  % Set up the termination function.
  function proj = proj_dh(a, b)
    I1 = (a == x_l); I2 = (a == x_u);
    proj = b;
    proj(I1) = min(0, proj(I1));
    proj(I2) = max(0, proj(I2));
  end
  function proj = proj_NKt(a, b)
    I0 = (a == 0);
    I1 = (a > 0);
    proj = b;
    proj(I0) = min(0, proj(I0));
    proj(I1) = 0;
  end
  o_at_x0 = copy(oracle);
  o_at_x0.eval(hparams.x0);
  g0 = hparams.constr_fn(hparams.x0);
  rho = global_tol * (1 + hparams.norm_fn(o_at_x0.grad_f_s()));
  eta = global_tol * (1 + hparams.norm_fn(g0 - hparams.set_projector(g0)));
  term_wrap = @(x,p) ...
    termination_check(x, p, o_at_x0, hparams.constr_fn, hparams.grad_constr_fn, ...
                      @proj_dh, @proj_NKt, hparams.norm_fn, rho, eta);

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
  ncvx_qc_qp.time_limit = 6000;
  
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
  base_hparam.termination_fn = term_wrap;
  
  % Create the IAPIAL hparams.
  ipla_hparam = base_hparam;
  ipla_hparam.acg_steptype = 'variable';
  ipla_hparam.sigma_min = sigma_min;

  % Run a benchmark test and print the summary.
  hparam_arr = {ipla_hparam};
  name_arr = {'IPL_A'};
  framework_arr = {@IAIPAL};
  solver_arr = {@ECG};
  
  % Run the test.
  [summary_tables, summary_mdls] = ...
    run_CCM_benchmark(ncvx_qc_qp, framework_arr, solver_arr, hparam_arr, name_arr);
  
  % Set up HiAPeM options.
  o_at_x0 = copy(oracle);
  o_at_x0.eval(hparams.x0);
  feas_at_x0 = feasibility(hparams.x0);
  rel_tol = min([...
    global_tol * (1 + hparams.norm_fn(o_at_x0.grad_f_s())), ...
    global_tol * (1 + feas_at_x0)]);
  opts = struct();
  opts.x0 = hparams.x0;
  opts.Lip0 = max([hparams.m, hparams.M]);
  opts.tol = rel_tol;
  opts.maxit = 1000000;
  opts.inc = 2;
  opts.dec= 2.5;
  opts.sig = 3;
  opts.maxsubit = 1000000;
  opts.rho = hparams.m; % Lower curvature.
  x_l_vec = ones(dimN, 1) * hparams.x_l;
  x_u_vec = ones(dimN, 1) * hparams.x_u;
  
  % Run the HiAPeM code.
  [x_hpm, ~, out_hpm] = HiAPeM_qcqp(...
    hparams.Q, hparams.c, hparams.d, dimM, x_l_vec, x_u_vec, opts);
  HPM_ratio = out_hpm.acg_ratio;
  o_at_x_hpm = copy(oracle);
  o_at_x_hpm.eval(x_hpm);
  fval_hpm = o_at_x_hpm.f_s();
  iter_hpm = out_hpm.niter;
  disp(['Total number of iterations is ', num2str(iter_hpm)])
  disp(['Function value is ', num2str(fval_hpm)])
  disp(['Feasibility is ', num2str(feasibility(x_hpm))]);
  
  %Aggregate
  o_tbl = agg_tbl(summary_tables, summary_mdls, iter_hpm, HPM_ratio);
  
  % Compute feasibility
  function feas = feasibility(x)
    m_cx = -hparams.constr_fn(x);
    m_cxp = hparams.set_projector(m_cx);
    feas = norm(m_cxp - m_cx);
  end

  function o_tbl = ...
      agg_tbl(summary_tbls, summary_mdls, iter_HiAPeM, HPM_ratio)
    ratio_IPL_A = summary_mdls.IPL_A.history.acg_ratio;
    o_tbl = [...
      table(sigma_min, dimN, x_l, x_u), summary_tbls.pdata, summary_tbls.iter, ... 
      table(iter_HiAPeM), table(ratio_IPL_A, HPM_ratio)];
  end

end

