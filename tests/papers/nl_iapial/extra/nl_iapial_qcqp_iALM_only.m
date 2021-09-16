% Solve a multivariate nonconvex quadratically constrained quadratic programming  
% problem constrained to a box using MULTIPLE SOLVERS.
run('../../../../init.m');

% Run an instance via the command line.
print_tbls(n);

%% Utility functions
function print_tbls(dimN) 

  % Initialize
  seed = 77777;
  dimM = 5;
  global_tol = 1e-5;
  m_vec = [1e2, 1e3, 1e4];
  M_vec = [1e4, 1e5, 1e6];
  r_vec = [5, 10, 20];
  first_tbl = true;

  % Variable M.
  m = 1e0;
  r = 1;
  for M=M_vec
    tbl_row = run_experiment(M, m, dimM, dimN, -r, r, seed, global_tol);
    if first_tbl
      o_tbl = tbl_row;
      first_tbl = false;
    else
      o_tbl = [o_tbl; tbl_row];
    end
  end
  
  % Variable m.
  M = 1e6;
  r = 1;
  for m=m_vec
    tbl_row = run_experiment(M, m, dimM, dimN, -r, r, seed, global_tol);
    if first_tbl
      o_tbl = tbl_row;
      first_tbl = false;
    else
      o_tbl = [o_tbl; tbl_row];
    end
  end
  
  % Variable r.
  m = 1e0;
  M = 1e6;
  for r=r_vec
    tbl_row = run_experiment(M, m, dimM, dimN, -r, r, seed, global_tol);
    if first_tbl
      o_tbl = tbl_row;
      first_tbl = false;
    else
      o_tbl = [o_tbl; tbl_row];
    end
  end
  
  disp(['Tables for dimN = ', num2str(dimN)]);
  disp(o_tbl);
  
end
function o_tbl = run_experiment(M, m, dimM, dimN, x_l, x_u, seed, global_tol)

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
  
  % Create the complicated iALM hparams.
  ialm_hparam = base_hparam;
  ialm_hparam.i_ineq_constr = true;
  ialm_hparam.rho0 = hparams.m;
  ialm_hparam.L0 = max([hparams.m, hparams.M]);
  ialm_hparam.rho_vec = hparams.m_constr_vec;
  ialm_hparam.L_vec = hparams.L_constr_vec;
  % Note that we are using the fact that |X|_F <= 1 over the eigenbox.
  ialm_hparam.B_vec = hparams.K_constr_vec;
  
  % Run a benchmark test and print the summary.
  hparam_arr = {ialm_hparam};
  name_arr = {'iALM'};
  framework_arr = {@iALM};
  solver_arr = {@ECG};
  
  % Run the test.
  [summary_tables, ~] = ...
    run_CCM_benchmark(...
      ncvx_qc_qp, framework_arr, solver_arr, hparam_arr, name_arr);
    
  %Aggregate
  o_tbl = agg_tbl(summary_tables);
  disp(o_tbl);

  function o_tbl = agg_tbl(summary_tbls)
    o_tbl = [...
      table(dimN, x_l, x_u), summary_tbls.pdata, summary_tbls.fval, ...
      summary_tbls.iter, summary_tbls.runtime, summary_tbls.mdata];
  end

end

