% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [model, history] = ACG(oracle, params)
% An accelerated composite gradient (ACG) algorithm for use inside of the accelerated inexact proximal point (AIPP) method.
%
% See Also:
%
%   **src/solvers/AIPP.m**
%
% Note:
%
%   Its iterates are generated according the paper:
%
%   Monteiro, R. D., Ortiz, C., & Svaiter, B. F. (2016). An adaptive accelerated first-order method for convex optimization.
%   *Computational Optimization and Applications*, 64(1), 31-73.
%
% Arguments:
%
%   oracle (Oracle): The oracle underlying the optimization problem.
%
%   params (struct): Contains instructions on how to call the algorithm. This should ideally be customized from by caller of this
%   algorithm, rather than the user.
%
% Returns:
%
%   A pair of structs containing model and history related outputs of the solved problem associated with the oracle and input
%   parameters.
%

  % Set some ACG global tolerances.
  INEQ_TOL = 1e-6;
  CURV_TOL = 1e-6;

  %% PRE-PROCESSING

  % Set REQUIRED input params.
  x0 = params.x0;
  mu = params.mu;
  L_max = params.L;
  prod_fn = params.prod_fn;
  norm_fn = params.norm_fn;

  % Fill in OPTIONAL input params.
  params = set_default_params(params);

  % Set other input params.
  termination_type = params.termination_type;
  i_early_stop = false;
  iter = 1;
  fn_iter = 0;
  grad_iter = 0;
  L_grad_f_s_est = params.L_grad_f_s_est;
  local_L_est = params.L_est;
  x_prev = params.x_prev;
  y_prev = params.y_prev;
  A_prev = params.A_prev;
  
  % Check for logging requirements.
  if params.i_logging
    history.stationarity_iters = [];
    history.stationarity_grad_iters = [];
    history.stationarity_values = [];
    history.iteration_values = [];
    history.time_values = [];
  end
  
  % Solver params.
  t_start = tic;
  time_limit = params.time_limit;
  iter_limit = params.iter_limit;
  
  % Pull in some constants to save compute time (based on the caller).
  if strcmp(termination_type, "aicg")
    phi_at_Z0 = params.phi_at_Z0;
    f1_at_Z0 = params.f1_at_Z0;
    Dg_Pt_grad_f1_at_Z0_Q = params.Dg_Pt_grad_f1_at_Z0_Q;
    aicg_M1 = params.aicg_M1;
    X0_mat = params.X0_mat;
    lambda = params.lambda;
    sigma = params.sigma;
    tau = sigma ^ 2 * (norm_fn(X0_mat) ^ 2 - norm_fn(x0) ^ 2);
  elseif strcmp(termination_type, "d_aicg")
    X0_mat = params.X0_mat;
    lambda = params.lambda;
    sigma = params.sigma;
    tau = sigma ^ 2 * (norm_fn(X0_mat) ^ 2 - norm_fn(x0) ^ 2);
  elseif strcmp(termination_type, "gd")
    tau = params.tau;
    theta = params.theta;
  elseif (any(strcmp(termination_type, {'aipp', 'aipp_sqr', 'adap_aidal'})))
    sigma = params.sigma;
  elseif strcmp(termination_type, "aipp_phase2")
    lambda = params.lambda;
    epsilon_bar = params.epsilon_bar;
  elseif strcmp(termination_type, "apd")
    sigma = params.sigma;
    theta = params.theta;
  elseif strcmp(termination_type, "none")
    % Do nothing, but check the validity.
  else
    % Unknown termination type.
    error(['Unknown termination conditions for termination_type = ', termination_type]);
  end
  
  % Initialize constants related to Gamma.
  if (strcmp(params.eta_type, 'recursive'))
    Gamma_at_x_prev = 0;
    grad_Gamma_at_x_prev = zeros(size(x0));
  elseif (strcmp(params.eta_type, 'accumulative'))
    scSum = 0;
    svSum = zeros(size(x0), 'like', x0);
    snSum = 0;
  elseif strcmp(params.eta_type, 'none')
    % Do nothing.
  else
    error('Unknown eta type!');
  end
    
  % Check if we should use variable stepsize approach.
  if strcmp(params.acg_steptype, "variable")
    L = mu + (local_L_est - mu) * params.init_mult_L;
  elseif strcmp(params.acg_steptype, "constant")
    L = L_max;
  else
      error('Unknown ACG steptype!');
  end
  
  % Safeguard against the case where (L == mu), i.e. lamK = Inf.
  L = max(L, mu + CURV_TOL);
  
  % Set up the oracle at x0
  o_x0 = oracle.eval(x0);
  f_at_x0 = o_x0.f_s() + o_x0.f_n();
  fn_iter = fn_iter + 1;
  
  %% LOCAL FUNCTIONS
  
  % Compute an estimate of L (and other quantities) based on two points.
  function [local_L_est, aux_struct] = compute_approx_iter(L, mu, A_prev, y_prev, x_prev)

    % Simple quantities.
    lamK = lamK_fn(mu, L);
    tauK = (1 + mu * A_prev) * lamK;
    a_prev = (tauK + sqrt(tauK ^ 2 + 4 * tauK * A_prev)) / 2;
    A = A_prev + a_prev;
    x_tilde_prev = (A_prev / A) * y_prev + a_prev / A * x_prev;

    % Oracle at x_tilde_prev.
    o_x_tilde_prev = oracle.eval(x_tilde_prev);
    f_s_at_x_tilde_prev = o_x_tilde_prev.f_s();
    grad_f_s_at_x_tilde_prev = o_x_tilde_prev.grad_f_s();
    
    % Oracle at y.
    y_prox_mult = lamK / (1 + lamK * mu);
    y_prox_ctr = x_tilde_prev - y_prox_mult * grad_f_s_at_x_tilde_prev;
    [y, o_y] = get_y(y_prox_ctr, y_prox_mult);
    f_s_at_y = o_y.f_s();
    
    % Estimate of L based on y and x_tilde_prev.
    LHS = f_s_at_y - (f_s_at_x_tilde_prev + prod_fn(grad_f_s_at_x_tilde_prev, y - x_tilde_prev));
    dist_xt_y = norm_fn(y - x_tilde_prev);
    RHS = L * dist_xt_y ^ 2 / 2;
    local_L_est = max(0, 2 * LHS / dist_xt_y ^ 2);    
    
    % Save auxiliary quantities.
    aux_struct.LHS = LHS;
    aux_struct.RHS = RHS;
    aux_struct.dist_xt_y = dist_xt_y;
    aux_struct.descent_cond = (LHS <= RHS + INEQ_TOL);
    aux_struct.y = y;
    aux_struct.o_y = o_y;
    aux_struct.o_x_tilde_prev = o_x_tilde_prev;
    aux_struct.lamK = lamK;
    aux_struct.tauK = tauK;
    aux_struct.a_prev = a_prev;
    aux_struct.A = A;
    aux_struct.x_tilde_prev = x_tilde_prev;
    aux_struct.f_s_at_x_tilde_prev = f_s_at_x_tilde_prev;
    aux_struct.grad_f_s_at_x_tilde_prev = grad_f_s_at_x_tilde_prev;
  end

  % Function for efficiently obtaining y.
  function [y, o_y] = get_y(prox_ctr, prox_mult)
    o_y_prox = oracle.eval(prox_ctr);
    if not(isfield(o_y_prox, 'f_s_at_prox_f_n') && isfield(o_y_prox, 'f_n_at_prox_f_n') && isfield(o_y_prox, 'grad_f_s_at_prox_f_n'))
         y = o_y_prox.prox_f_n(prox_mult);
         o_y = oracle.eval(y);
    end
    if(isfield(o_y_prox, 'f_s_at_prox_f_n'))
      o_y.f_s = @() o_y_prox.f_s_at_prox_f_n(prox_mult);
    end
    if(isfield(o_y_prox, 'f_n_at_prox_f_n'))
      o_y.f_n = @() o_y_prox.f_n_at_prox_f_n(prox_mult);
    end
    if(isfield(o_y_prox, 'grad_f_s_at_prox_f_n'))
      o_y.grad_f_s = @() o_y_prox.grad_f_s_at_prox_f_n(prox_mult);
    end
  end
   
  %% MAIN ALGORITHM

  while true
    
    % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit && i_early_stop)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit && i_early_stop)
      break; 
    end
        
    %% COMPUTE y AND L ADAPTIVELY + OTHER KEY VARIABLES.
        
    % Variable L updates.
    if strcmp(params.acg_steptype, "variable")
      
      % Compute L_est and auxiliary quantities.
      L = max(L, mu);
      [~, aux_struct] = compute_approx_iter(L, mu, A_prev, y_prev, x_prev);
      iter = iter + 1;
      fn_iter = fn_iter + 1;
      grad_iter = grad_iter + 1;
      
      % Update based on the value of the local L compared to the current estimate of L.
      while (~aux_struct.descent_cond)        
        L = min(L_max, L * params.mult_L);
        [~, aux_struct] = compute_approx_iter(L, mu, A_prev, y_prev, x_prev);
        iter = iter + 1;
        fn_iter = fn_iter + 1;
        grad_iter = grad_iter + 1;
        
        % Additional APD descent condition.
        if strcmp(termination_type, 'apd')
          % Parse the computed iterates.
          A = aux_struct.A;
          y = aux_struct.y;
          lamK = aux_struct.lamK;
          a_prev = aux_struct.a_prev;
          f_s_at_x_tilde_prev = aux_struct.f_s_at_x_tilde_prev;
          grad_f_s_at_x_tilde_prev = aux_struct.grad_f_s_at_x_tilde_prev;
          x_tilde_prev = aux_struct.x_tilde_prev;
          dist_xt_y = aux_struct.dist_xt_y;
          o_x_tilde_prev = aux_struct.o_x_tilde_prev;
          o_y = aux_struct.o_y;
        end
        
        if (L >= L_max && ~aux_struct.descent_cond)
          error('Theoretical upper bound on the upper curvature L_max does not appear to be correct!');
        end
        if (toc(t_start) > time_limit)
          break;
        end
      end
      
      % Load auxiliary quantities
      y = aux_struct.y;
      o_y = aux_struct.o_y;
      lamK = aux_struct.lamK;
      tauK = aux_struct.tauK;
      a_prev = aux_struct.a_prev;
      A = aux_struct.A;
      x_tilde_prev = aux_struct.x_tilde_prev;
      f_s_at_x_tilde_prev = aux_struct.f_s_at_x_tilde_prev;
      grad_f_s_at_x_tilde_prev = aux_struct.grad_f_s_at_x_tilde_prev;
      dist_xt_y = aux_struct.dist_xt_y;
      
    % Constant L updates.
    elseif strcmp(params.acg_steptype, "constant")
            
      % Iteration parameters.
      lamK = lamK_fn(mu, L);
      tauK = (1 + mu * A_prev) * lamK;
      a_prev = (tauK + sqrt(tauK ^ 2 + 4 * tauK * A_prev)) / 2;
      A = A_prev + a_prev;
      x_tilde_prev = (A_prev / A) * y_prev + (a_prev / A) * x_prev;

      % Oracle at x_tilde_prev.
      o_x_tilde_prev = oracle.eval(x_tilde_prev); 
      f_s_at_x_tilde_prev = o_x_tilde_prev.f_s();
      grad_f_s_at_x_tilde_prev = o_x_tilde_prev.grad_f_s();
      fn_iter = fn_iter + 1;
      grad_iter = grad_iter + 1;
      
      % Oracle at y.
      y_prox_mult = lamK / (1 + lamK * mu);
      y_prox_ctr = x_tilde_prev - y_prox_mult * grad_f_s_at_x_tilde_prev;
      [y, o_y] = get_y(y_prox_ctr, y_prox_mult);
      
      % Update the counter.
      iter = iter + 1;
    
    else 
      error('Unknown steptype!');
    end   
    
    f_s_at_y = o_y.f_s();
    f_n_at_y = o_y.f_n();
    f_at_y = f_s_at_y + f_n_at_y;
    
    %% COMPUTE (u, η), Γ, and x.
           
    % Compute x and other helper variables.
    x = 1 / (1 + mu * A) * (x_prev  - a_prev / lamK * (x_tilde_prev - y) + mu * (A_prev * x_prev + a_prev * y));
    if strcmp(termination_type, "apd")
      y_xT = y - x_tilde_prev;
      r = -(L + mu) * y_xT + o_y.grad_f_s() - grad_f_s_at_x_tilde_prev;
      grad_iter = grad_iter + 1;
      norm_y_xT = norm_fn(y_xT);
      y_y0 = y - x0;
      norm_y_y0 = norm_fn(y_y0);
      u = r;
    else
      u = (x0 - x) / A;
    end
        
    % Add metrics for the current outer iteration if needed.
    if params.i_logging
      if (isfield(params, "stationarity_fn"))
        history.stationarity_values(end + 1) = params.stationarity_fn(u, y);
        history.stationarity_iters(end + 1) = params.base_iter + iter;
        history.stationarity_grad_iters(end + 1) = params.base_grad_iter + grad_iter;
      end
      history.iteration_values(end + 1) = iter;
      history.time_values(end + 1) = toc(t_start);
    end

    % Compute eta.
    if strcmp(params.eta_type, 'recursive')
      % Recursive
      gamma_at_x = f_n_at_y + f_s_at_x_tilde_prev + prod_fn(grad_f_s_at_x_tilde_prev, y - x_tilde_prev) + ...
                   (mu / 2) * norm_fn(y - x_tilde_prev) ^ 2 + 1 / lamK * prod_fn(x_tilde_prev - y, x - y) + ...
                   (mu / 2) * norm_fn(x - y) ^ 2;
      Gamma_at_x = a_prev / A * gamma_at_x + A_prev / A * Gamma_at_x_prev + A_prev / A * prod_fn(grad_Gamma_at_x_prev, x - x_prev) + ...
                   (mu / 2) * A_prev / A * norm_fn(x - x_prev) ^ 2;
      eta_rcr = f_at_y - Gamma_at_x - prod_fn(u, y - x);
      fn_iter = fn_iter + 1;
      eta = eta_rcr;
    elseif strcmp(params.eta_type, 'accumulative')
      % Accumulative
      p_at_y = f_s_at_x_tilde_prev + prod_fn(grad_f_s_at_x_tilde_prev, y - x_tilde_prev) + mu / 2 * norm_fn(y - x_tilde_prev) ^ 2 + f_n_at_y;
      sci = p_at_y + mu / 2 * norm_fn(y) ^ 2 - (1 / lamK) * prod_fn(y, x_tilde_prev - y);
      svi = - mu * y + (1 / lamK) * (x_tilde_prev - y);
      sni = mu / 2;
      scSum = scSum + a_prev * sci;
      svSum = svSum + a_prev * svi;
      snSum = snSum + a_prev * sni;
      Gamma = @(xG) (scSum + prod_fn(svSum, xG) + snSum * norm_fn(xG) ^ 2) / A;
      Gamma_at_x = Gamma(x);
      eta_acc = f_at_y - Gamma_at_x - prod_fn(u, y - x);
      fn_iter = fn_iter + 1;
      eta = eta_acc;
    elseif strcmp(params.eta_type, 'none')
      eta = 0;
    else
      error('Unknown eta type!');
    end
    
    % NOTE: Gamma minorizes the function f := f_s + f_n.
        
    % Compute eta.
    exact_eta = eta;
    eta = max([0, exact_eta]);
    % Check the negativity of eta in a relative sense.
    if strcmp(termination_type, "aipp")
      relative_exact_eta = exact_eta / max([norm_fn(u + x0 - y) ^ 2 / 2, 0.01]);
      if (relative_exact_eta < -INEQ_TOL)
        error(['eta is negative with a value of ', num2str(exact_eta)]);
      end
    end
    
    %% CHECK EARLY STATIONARITY.
    if (any(strcmp(termination_type, {'apd'})))
      v = params.stationarity_fn(u, y);
      if (norm_fn(v) <= params.opt_tol)
        i_early_stop = true;
        break
      end
    end
        
    %% CHECK INVARIANTS.
    
    % Sufficient descent.
    if (any(strcmp(termination_type, {'gd'})))
      large_gd = f_at_x0;
      small_gd = f_at_y + prod_fn(u, x0 - y) - eta;
      del_gd = large_gd - small_gd;
      base = max([abs(large_gd), abs(small_gd), 0.01]);
      if (del_gd / base < -INEQ_TOL)
        model.status = -1;
        break;
      end
    end
    
    % Minorization.
    if (any(strcmp(termination_type, {'gd', 'adap_aidal'})))
      small_gd = norm_fn(A * u + y - x0) ^ 2 + 2 * A * eta;
      large_gd = norm_fn(y - x0) ^ 2;
      del_gd = large_gd - small_gd;
      base = max([abs(large_gd), abs(small_gd), 0.01]);
      if (del_gd / base < -INEQ_TOL)
        model.status = -2;
        break;
      end
    end
    
    if (any(strcmp(termination_type, {'aicg', 'd_aicg'})))
      u2 = u + mu * (y - x);
      eta2 = eta + mu * norm_fn(y - x) ^ 2 / 2;
      small_gd = (1 / (1 + mu * A)) * norm_fn(A * u2 + y - x0) ^ 2 + 2 * A * eta2;
      large_gd = norm_fn(y - x0) ^ 2;
      del_gd = large_gd - small_gd;
      base = max([abs(large_gd), abs(small_gd), 0.01]);
      if (del_gd / base < -INEQ_TOL)
        model.status = -2;
        break;
      end
    end
    
    if (any(strcmp(termination_type, {'apd'})))
      large_gd = f_at_x0;
      small_gd = f_at_y - prod_fn(r, y_y0);
      del_gd = large_gd - small_gd;
      base = max([abs(large_gd), abs(small_gd), 0.01]);
      if (del_gd / base < -INEQ_TOL)
        model.status = -1;
        break;
      end
      if (mu * A * norm_y_xT ^2 > norm_y_y0 ^2)
        model.status = -2;
        break;
      end
    end
    
    %% CHECK FOR TERMINATION.

    % Update status when we have a possible triple for termination. (NEW FIX)
    i_early_stop = true;
    
    % Termination for the AIPP method (Phase 1).
    if strcmp(termination_type, "aipp")
      if (norm_fn(u) ^ 2 + 2 * eta <= sigma * norm_fn(x0 - y + u) ^ 2 + INEQ_TOL)
        break;
      end
      
    % Termination for the AIPP method (with sigma square).
    elseif (any(strcmp(termination_type, {'aipp_sqr', 'adap_aidal'})))
      if (norm_fn(u) ^ 2 + 2 * eta <= sigma ^ 2 * norm_fn(x0 - y + u) ^ 2 + INEQ_TOL)
        break;
      end
      
    % Termination for the AIPP method (Phase 2).
    elseif strcmp(termination_type, "aipp_phase2")
      if (eta <= lambda * epsilon_bar)
        break;
      end
           
    % Termination for the R-AIPP method.
    elseif strcmp(termination_type, "gd")
      phi_tilde_at_y = f_at_y - 1 / 2 * norm_fn(y - x0) ^ 2;
      phi_tilde_at_x0 = f_at_x0;
      cond1 = (norm_fn(x0 - y + u) ^ 2 <= theta * (phi_tilde_at_x0 - phi_tilde_at_y));
      cond2 = (2 * L_grad_f_s_est * eta <= tau * norm_fn(x0 - y + u) ^ 2);
      if (cond1 && cond2)
        break;
      end
      
    % Termination for the AICG method.
    elseif strcmp(termination_type, "aicg")
      % Helper variables
      u2 = u + mu * (x - y);
      eta2 = eta + mu * norm_fn(y - x) ^ 2 / 2;
      phi_at_Z_approx = f1_at_Z0 + o_y.orig_f2_s() + o_y.orig_f_n() + prod_fn(Dg_Pt_grad_f1_at_Z0_Q, y - x0) + ...
                        aicg_M1 / 2 * norm_fn(y - x0) ^ 2;
      fn_iter = fn_iter + 1;
      delta_phi = phi_at_Z0 - phi_at_Z_approx;
      % Main condition checks
      cond1 = (norm_fn(u2) ^ 2 <= 4 * lambda * delta_phi);
      cond2 = (2 * eta2 <= 4 * lambda * delta_phi);
      cond3 = (norm_fn(y - x0) + tau <= 4 * lambda * delta_phi);
      if (cond1 && cond2 && cond3)
        model.phi_at_Z_approx = phi_at_Z_approx;
        break;
      end
      
    % Termination for the D-AICG method.
    elseif strcmp(termination_type, "d_aicg")
      u2 = u + mu * (x - y);
      eta2 = eta + mu * norm_fn(y - x) ^ 2 / 2;
      if (norm_fn(u2) ^ 2  + 2 * eta2 <= sigma ^ 2 * norm_fn(y - x0) + tau)
        break;
      end
      
    % Termination for the APD method.
    elseif strcmp(termination_type, "apd")
      cond1 = norm_fn(r - y_y0)^2 <= theta * (f_at_x0 - f_at_y + norm_y_y0^2 / 2);
      cond2 = norm_fn(r) <= sigma * norm_y_y0;
      if (cond1 && cond2)
        break;
      end
    end
    
    %% UPDATE VARIABLES.
       
    % Update iterates.
    if (strcmp(params.eta_type, 'recursive'))
      grad_gamma_at_x = 1 / lamK * (x_tilde_prev - y) + mu * (x - y);
      grad_Gamma_at_x = a_prev / A * grad_gamma_at_x + A_prev / A * grad_Gamma_at_x_prev + A_prev / A * mu * (x - x_prev);
      Gamma_at_x_prev = Gamma_at_x;
      grad_Gamma_at_x_prev = grad_Gamma_at_x;
    end
    A_prev = A;
    y_prev = y;
    x_prev = x;
           
  end
  
  %% POST-PROCESSING.
  
  % Successful stop.
  if (i_early_stop) 
    model.status = 1;
    model.y = y;
    model.o_y = o_y;
    model.f_s_at_y = f_s_at_y;
    model.x = x;
    model.x0 = x0;
    model.u = u;
    model.eta = eta;
    model.A = A;
    % Other modifications
    if (any(strcmp(termination_type, {'aicg', 'd_aicg'})))
      model.u = u + mu * (y - x);
      model.eta = eta + mu * norm_fn(y - x) ^ 2 / 2;
    end
  end
  
  % Other outputs.
  model.L_est = L;
  history.iter = iter;
  history.fn_iter = fn_iter;
  history.grad_iter = grad_iter;
  history.runtime = toc(t_start);
          
end

%% HELPER FUNCTIONS

function out_lamK =lamK_fn(mu, L)
  out_lamK = 1 / (L - mu);
end

% Fills in parameters that were not set as input.
function params = set_default_params(params)

  % Overwrite if necessary.
  if (~isfield(params, 'acg_steptype'))
    params.acg_steptype = 'constant';
  end
  if (~isfield(params, 'termination_type'))
    params.termination_type = 'none';
  end
  if (~isfield(params, 'eta_type'))
    params.eta_type = 'recursive';
  end
  if (~isfield(params, 'L_grad_f_s_est'))
    params.L_grad_f_s_est = params.L;
  end
  if (~isfield(params, 'L_est'))
    params.L_est = params.L;
  end
  if (~isfield(params, 'mult_L'))
    params.mult_L = 1.25;
  end
  if (~isfield(params, 'init_mult_L'))
    params.init_mult_L = 0.5;
  end
  if (~isfield(params, 'z0'))
    params.z0 = params.x0;
  end
  if (~isfield(params, 'x_prev'))
    params.x_prev = params.x0;
  end
  if (~isfield(params, 'y_prev'))
    params.y_prev = params.x0;
  end
  if (~isfield(params, 'A_prev'))
    params.A_prev = 0;
  end

end
