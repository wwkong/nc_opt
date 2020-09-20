%{

FILE DATA
---------
Last Modified: 
  August 2, 2020
Coders: 
  Weiwei Kong, Jiaming Liang

%}

function [model, history] = ACG(oracle, params) 
% An accelerated composite gradient (ACG) algorithm for use inside of the
% accelerated inexact proximal point (AIPP) method.
% 
% See Also:
%  
%   **src.solvers.AIPP**
%
% Note:
% 
%   Its iterates are generated according the paper:
%
%   Monteiro, R. D., Ortiz, C., & Svaiter, B. F. (2016). An adaptive 
%   accelerated first-order method for convex optimization. *Computational 
%   Optimization and Applications*, 64(1), 31-73.
%
% Arguments:
%
%   oracle (Oracle): The oracle underlying the optimization problem.
%
%   params (struct): Contains instructions on how to call the algorithm. This 
%     should ideally  be customized from by caller of this algoirthm, rather 
%     than the user.
%
% Returns: 
%   
%   A pair of structs containing model and history related outputs of the 
%   solved problem associated with the oracle and input parameters.
%

  % Set some ACG global tolerances.
  INEQ_COND_ERR_TOL = 1e-6;
  CURV_TOL = 1e-6;

  % -----------------------------------------------------------------------
  %% PRE-PROCESSING
  % -----------------------------------------------------------------------
    
  % Set REQUIRED input params.
  x0 = params.x0;
  mu = params.mu;
  L_max = params.L;
  prod_fn = params.prod_fn;
  norm_fn = params.norm_fn;
  termination_type = params.termination_type;
  
  % Fill in OPTIONAL input params.
  params = set_default_params(params);
  
  % Set other input params.
  status = 0;
  iter = 1;
  t_start = params.t_start;
  L_grad_f_s_est = params.L_grad_f_s_est;
  L_est = params.L_est;
  x_prev = params.x_prev;
  y_prev = params.y_prev;
  A_prev = params.A_prev;
  
  % Solver params.
  time_limit = params.time_limit;
  iter_limit = params.iter_limit;
  
  % Pull in some constants to save compute time (based on the caller).
  if (termination_type == "aicg")
    phi_at_Z0 = params.phi_at_Z0;
    f1_at_Z0 = params.f1_at_Z0;
    Dg_Pt_grad_f1_at_Z0_Q = params.Dg_Pt_grad_f1_at_Z0_Q;
    aicg_M1 = params.aicg_M1;
    X0_mat = params.X0_mat;
    lambda = params.lambda;
    sigma = params.sigma;
    tau = sigma ^ 2 * (norm_fn(X0_mat) ^ 2 - norm_fn(x0) ^ 2);
  elseif (termination_type == "d_aicg")
    X0_mat = params.X0_mat;
    lambda = params.lambda;
    sigma = params.sigma;
    tau = sigma ^ 2 * (norm_fn(X0_mat) ^ 2 - norm_fn(x0) ^ 2);
  elseif (termination_type == "gd")
    tau = params.tau;
    theta = params.theta;
  elseif (any(strcmp(termination_type, {'aipp', 'aipp_sqr'})))
    sigma = params.sigma;
  elseif (termination_type == "aipp_phase2")
    lambda = params.lambda;
    epsilon_bar = params.epsilon_bar;
  end
  
  % Initialize constants related to Gamma.
  if (strcmp(params.eta_type, 'recursive'))
    Gamma_at_x_prev = 0;
    grad_Gamma_at_x_prev = zeros(size(x0));
  elseif (strcmp(params.eta_type, 'accumulative'))
    scSum = 0;
    svSum = zeros(size(x0), 'like', x0);
    snSum = 0;
  else
    error('Unknown eta type!');
  end
    
  % Check if we should use variable stepsize approach.
  if (params.acg_steptype == "variable")
    if (~isfield(params, 'mult_L'))
      mult_L = 1.25; % Default multiplier
    else  
      mult_L = params.mult_L;
    end
    L = mu + (L_est - 1) / 100;
  elseif (params.acg_steptype == "constant")
    L = L_max;
  else
      error('Unknown ACG steptype!');
  end
  
  % Safeguard against the case where (L == mu), i.e. nu = Inf.
  L = max(L, mu + CURV_TOL);
  
  % Set up the oracle at x0
  o_x0 = oracle.eval(x0);
  f_at_x0 = o_x0.f_s() + o_x0.f_n();
  
  % ---------------------------------------------------------------------
  %% LOCAL FUNCTIONS
  % ---------------------------------------------------------------------
  
  % Compute an estimate of L based on two points.
  function L_est  = compute_L_est(L, mu, A_prev, y_prev, x_prev)

    % Simple quantities.
    nu = nu_fn(mu, L);
    nu_prev = (1 + mu * A_prev) * nu;
    a_prev = (nu_prev + sqrt(nu_prev ^ 2 + 4 * nu_prev * A_prev)) / 2;
    A = A_prev + a_prev;
    x_tilde_prev = (A_prev / A) * y_prev + a_prev / A * x_prev;

    % Oracle at x_tilde_prev.
    o_x_tilde_prev = oracle.eval(x_tilde_prev);
    f_s_at_x_tilde_prev = o_x_tilde_prev.f_s();
    grad_f_s_at_x_tilde_prev = o_x_tilde_prev.grad_f_s();
    
    % Oracle at y.
    y_prox_mult = nu / (1 + nu * mu);
    y_prox_ctr = x_tilde_prev - y_prox_mult * grad_f_s_at_x_tilde_prev;
    [y, o_y] = get_y(y_prox_ctr, y_prox_mult);
    f_s_at_y = o_y.f_s();
    
    % Estimate of L based on y and x_tilde_prev.
    L_est = ...
      max(0, 2 * (f_s_at_y - ...
        (f_s_at_x_tilde_prev + ...
         prod_fn(grad_f_s_at_x_tilde_prev, y - x_tilde_prev))) / ...
      norm_fn(y - x_tilde_prev) ^ 2);
  end

  % Function for efficiently obtaining y.
  function [y, o_y] = get_y(prox_ctr, prox_mult)
    o_y_prox = oracle.eval(prox_ctr);
    if not(isfield(o_y_prox, 'f_s_at_prox_f_n') && ...
           isfield(o_y_prox, 'f_n_at_prox_f_n') && ...
           isfield(o_y_prox, 'grad_f_s_at_prox_f_n'))
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
   
  % -----------------------------------------------------------------------
  %% MAIN ALGORITHM
  % -----------------------------------------------------------------------
  while true
    
    % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit)
      break;
    end
        
    % ---------------------------------------------------------------------
    %% COMPUTE y AND L ADAPTIVELY + OTHER KEY VARIABLES.
    % ---------------------------------------------------------------------
        
    % Variable L updates.
    if (params.acg_steptype == "variable")
      
      % Compute L_est.
      L = max(L, mu);
      L_est = compute_L_est(L, mu, A_prev, y_prev, x_prev);
      iter = iter + 1;
      
      % Loop if a violation occurs.
      while (L < min([L_est, L_max]))
        L = min(L_max, L_est * mult_L);
        L_est = compute_L_est(L, mu, A_prev, y_prev, x_prev);
        iter = iter + 1;
      end
      
    end
            
    % Iteration parameters.
    nu = nu_fn(mu, L);
    nu_prev = (1 + mu * A_prev) * nu;
    a_prev = (nu_prev + sqrt(nu_prev ^ 2 + 4 * nu_prev * A_prev)) / 2;
    A = A_prev + a_prev;
    x_tilde_prev = (A_prev / A) * y_prev + (a_prev / A) * x_prev;

    % Oracle at x_tilde_prev.
    o_x_tilde_prev = oracle.eval(x_tilde_prev); 
    f_s_at_x_tilde_prev = o_x_tilde_prev.f_s();
    grad_f_s_at_x_tilde_prev = o_x_tilde_prev.grad_f_s();

    % Oracle at y.
    y_prox_mult = nu / (1 + nu * mu);
    y_prox_ctr = x_tilde_prev - y_prox_mult * grad_f_s_at_x_tilde_prev;
    [y, o_y] = get_y(y_prox_ctr, y_prox_mult);
    f_s_at_y = o_y.f_s();
    f_n_at_y = o_y.f_n();
    f_at_y = f_s_at_y + f_n_at_y;
    
    % ---------------------------------------------------------------------
    %% COMPUTE (u, η), Γ, and x.
    % ---------------------------------------------------------------------
           
    % Compute x and u.
    x = 1 / (1 + mu * A) * ...
      (x_prev  - a_prev / nu * (x_tilde_prev - y) + ...
       mu * (A_prev * x_prev + a_prev * y));
    u = (x0 - x) / A;

    % Compute eta.
    if (strcmp(params.eta_type, 'recursive'))
      % -------- Gamma Function (recursive) --------
      gamma_at_x = ...
        f_n_at_y + ...
        f_s_at_x_tilde_prev + ...
        prod_fn(grad_f_s_at_x_tilde_prev, y - x_tilde_prev) + ...
        (mu / 2) * norm_fn(y - x_tilde_prev) ^ 2 + ...
        1 / nu * prod_fn(x_tilde_prev - y, x - y) + ...
        (mu / 2) * norm_fn(x - y) ^ 2;
      Gamma_at_x = ...
        a_prev / A * gamma_at_x + ...
        A_prev / A * Gamma_at_x_prev + ...
        A_prev / A * prod_fn(grad_Gamma_at_x_prev, x - x_prev) + ...
        (mu / 2) * A_prev / A * norm_fn(x - x_prev) ^ 2;
      eta_rcr = f_at_y - Gamma_at_x - prod_fn(u, y - x);
      eta = eta_rcr;
      % -------- End Gamma Function (recursive) --------   
    elseif (strcmp(params.eta_type, 'accumulative'))
      % -------- Gamma Function (accumulative) --------
      p_at_y = f_s_at_x_tilde_prev + ...
        prod_fn(grad_f_s_at_x_tilde_prev, y - x_tilde_prev) + ...
        mu / 2 * norm_fn(y - x_tilde_prev) ^ 2 + f_n_at_y;
      sci = p_at_y + mu / 2 * norm_fn(y) ^ 2 - (1 / nu) * ...
            prod_fn(y, x_tilde_prev - y);
      svi = - mu * y + (1 / nu) * (x_tilde_prev - y);
      sni = mu / 2;
      scSum = scSum + a_prev * sci;
      svSum = svSum + a_prev * svi;
      snSum = snSum + a_prev * sni;
      Gamma = @(xG) ...
        (scSum + prod_fn(svSum, xG) + snSum * norm_fn(xG) ^ 2) / A;
      Gamma_at_x = Gamma(x);
      eta_acc = f_at_y - Gamma_at_x - prod_fn(u, y - x);
      eta = eta_acc;
      % ------ End Gamma Function (accumulative) ------
    else
      error('Unknown eta type!');
    end
    
    % NOTE: Gamma minorizes the function f := f_s + f_n.
        
    % Compute eta.
    exact_eta = eta;
    eta = max([0, exact_eta]);
    % Check the negativity of eta in a relative sense.
    if (termination_type == "aipp")
      relative_exact_eta = exact_eta / (norm_fn(u + x0 - y) ^ 2 / 2);
      if (relative_exact_eta < -INEQ_COND_ERR_TOL)
        error(['eta is negative with a value of ', num2str(exact_eta)]);
      end
    end
        
    % ---------------------------------------------------------------------
    %% CHECK INVARIANTS.
    % ---------------------------------------------------------------------
        
    % Sufficient descent.
    if (any(strcmp(termination_type, {'gd'})))
      large_gd = f_at_x0;
      small_gd = f_at_y + prod_fn(u, x0 - y) - eta;
      del_gd = large_gd - small_gd;
      base = max([abs(large_gd), abs(small_gd), 0.01]);
      if (del_gd / base < -INEQ_COND_ERR_TOL)
        model.status = -1;
        break;
      end
    end
    
    % Minorization.
    if (any(strcmp(termination_type, {'gd', 'aicg', 'd_aicg'})))
      small_gd = norm_fn(A * u + y - x0) ^ 2 + 2 * A * eta;
      large_gd = norm_fn(y - x0) ^ 2;
      del_gd = large_gd - small_gd;
      base = max([abs(large_gd), abs(small_gd), 0.01]);
      if (del_gd / base < -INEQ_COND_ERR_TOL)
        model.status = -2;
        break;
      end
    end
    
    % ---------------------------------------------------------------------
    %% CHECK FOR TERMINATION.
    % ---------------------------------------------------------------------
    
    % Termination for the AIPP method (Phase 1).
    if (termination_type == "aipp")
      if (norm_fn(u) ^ 2 + 2 * eta <= sigma * norm_fn(x0 - y + u) ^ 2)
        status = 1;
        break;
      end
      
    % Termination for the AIPP method (with sigma square).
    elseif (termination_type == "aipp_sqr")
      if (norm_fn(u) ^ 2 + 2 * eta <= sigma ^ 2 * norm_fn(x0 - y + u) ^ 2)
        status = 1;
        break;
      end
      
    % Termination for the AIPP method (Phase 2).
    elseif (termination_type == "aipp_phase2")
      if (eta <= lambda * epsilon_bar)
        status = 1;
        break;
      end
           
    % Termination for the R-AIPP method.
    elseif (termination_type == "gd")
      phi_tilde_at_y = f_at_y - 1 / 2 * norm_fn(y - x0) ^ 2;
      phi_tilde_at_x0 = f_at_x0;
      cond1 = ...
        (norm_fn(x0 - y + u) ^ 2 <= ...
        theta * (phi_tilde_at_x0 - phi_tilde_at_y));
      cond2 = (2 * L_grad_f_s_est * eta <= tau * norm_fn(x0 - y + u) ^ 2);
      if (cond1 && cond2)
        status = 1;
        break;
      end
      
    % Termination for the AICG method.
    elseif (termination_type == "aicg")
      % Helper variables
      phi_at_Z_approx = ...
        f1_at_Z0 + o_y.orig_f2_s() + o_y.orig_f_n() + ...
        prod_fn(Dg_Pt_grad_f1_at_Z0_Q, y - x0) + ...
        aicg_M1 / 2 * norm_fn(y - x0) ^ 2;
      delta_phi = phi_at_Z0 - phi_at_Z_approx;
      % Main condition checks
      cond1 = ...
        (norm_fn(u) ^ 2 <= 4 * lambda * delta_phi);
      cond2 = ...
        (2 * eta <= 4 * lambda * delta_phi);
      cond3 = ...
        (norm_fn(y - x0) + tau <= 4 * lambda * delta_phi);
      if (cond1 && cond2 && cond3)
        model.phi_at_Z_approx = phi_at_Z_approx;
        status = 1;
        break;
      end
      
    % Termination for the D-AICG method.
    elseif (termination_type == "d_aicg")
      if (norm_fn(u) ^ 2  + 2 * eta <= sigma ^ 2 * norm_fn(y - x0) + tau)
        status = 1;
        break;
      end

    % Unknown termination.
    else
      error(...
        ['Unknown termination conditions for termination_type = ', ...
         termination_type]);      
    end
    
    % ---------------------------------------------------------------------
    %% UPDATE VARIABLES.
    % ---------------------------------------------------------------------
       
    % Update iterates.
    if (strcmp(params.eta_type, 'recursive'))
      grad_gamma_at_x = ...
        1 / nu * (x_tilde_prev - y) + mu * (x - y);
      grad_Gamma_at_x = ...
        a_prev / A * grad_gamma_at_x + ...
        A_prev / A * grad_Gamma_at_x_prev + ...
        A_prev / A * mu * (x - x_prev);
      Gamma_at_x_prev = Gamma_at_x;
      grad_Gamma_at_x_prev = grad_Gamma_at_x;
    end
    A_prev = A;
    y_prev = y;
    x_prev = x;
    iter = iter + 1;
           
  end
  
  % -----------------------------------------------------------------------
  %% POST-PROCESSING.
  % -----------------------------------------------------------------------
  
  % Successful stop.
  if (status == 1) 
    model.y = y;
    model.o_y = o_y;
    model.f_s_at_y = f_s_at_y;
    model.x = x;
    model.x0 = x0;
    model.u = u;
    model.eta = eta;
    model.A = A;
    model.L_est = L;
  end
  
  % Other outputs.
  model.status = status;
  history.iter = iter;
  history.runtime = toc(t_start);
          
end

% -------------------------------------------------------------------------
%% HELPER FUNCTIONS
% -------------------------------------------------------------------------

function out_nu = nu_fn(mu, L)
  out_nu = 1 / (L - mu);
end

% Fills in parameters that were not set as input.
function params = set_default_params(params)

  % Overwrite if necessary.
  if (~isfield(params, 'eta_type'))
    params.eta_type = 'recursive';
  end
  if (~isfield(params, 'L_grad_f_s_est'))
    params.L_grad_f_s_est = params.L;
  end
  if (~isfield(params, 'L_est'))
    params.L_est = params.L;
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