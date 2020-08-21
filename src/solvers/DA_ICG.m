%{

The doubly accelerated inexact composite gradient (D-AICG) method.

Coded by: Weiwei (William) Kong

VERSION 0.5
-----------

Input
------
spectral_oracle:
  An oracle function (see README.md for details).
params:
  Input parameters of this function (see README.md for details).

Output
------
mod:
  A struct array containing model related outputs.
hist:
  A struct array containing history (e.g. iteration counts, runtime, etc.)
  related outputs.

%}

function [model, history] = DA_ICG(spectral_oracle, params)
  
  % Timer start
  t_start = tic;
                            
  % ===========================================
  % --------------- Parse Input ---------------
  % ===========================================
   
  % Induced norm
  norm_fn = @(a) norm(a, 'fro');
  prod_fn = @(a,b) sum(dot(a, b));
  params.norm_fn = norm_fn;
  params.prod_fn = prod_fn;
  
  % Main params
  Z_y0 = params.z0; % This is the initial point of the algorithm
  m2 = params.m2;
  M1 = params.M1;
  M2 = params.M2;
  tol = params.tol;
  decomp_fn = params.decomp_fn;
  macheps_Delta = 1e-2;
  
  % Check for a projector function
  if isfield(params, 'Omega_projection')
    Omega_projection = params.Omega_projection;
  else
    % Replace with id if not found
    Omega_projection = @(X) X;
  end 
  
  % Check for a projector function
  if isfield(params, 'is_monotone')
    is_monotone = params.is_monotone;
  else
    is_monotone = false;
  end 

  % Convexify inputs
  o_spectral_oracle = spectral_oracle;
  if isfield(params, 'xi')
    xi = params.xi;
  else
    xi = M1;
  end
  spectral_oracle = ...
    @(x, sigma) ...
       redistribute_curvature(x, sigma, o_spectral_oracle, xi);
    
  % ========================================
  % --------------- Preamble ---------------
  % ========================================
  
  % Initialize basic parameters.
  [zM, zN] = size(Z_y0);
  zR = min(zM, zN);
  spo_at_Z_y0 = spectral_oracle(Z_y0, []);
  model.x = Z_y0;
  model.v = -Inf;
  history.iter = 0;
  if (M1 - xi <= 0)
    if isfield(params, 'lambda')
      lambda = params.lambda;
    else
      lambda = 5 / M1;
    end
  else
    lambda = 1 / (4 * M1);
  end
  
  % Check the stepsize strategy.
  if isfield(params, 'steptype')
    steptype = params.steptype;
  else
    steptype = 'constant';
  end
  
  % Set sigma.
  if ((1 / 2 - max([lambda * (M1 - xi), 0])) < 0)
    error('lambda is too large to set sigma!');
  end
  sigma = sqrt(1 / 2 - max([lambda * (M1 - xi), 0]));
    
  % Set curvature functions.
  mu_fn = @(lam) 1 / 2;
  L_prox_fn = @(lam, L) lam * L + 1; % prox curvature function
  
  % Set outer acceleration parameters.
  A0 = 0;
  Z_x0 = Z_y0;
  
  % Set parameters for the first ACG method call 
  % (and possibly subsequent calls).
  params_acg = params;
  params_acg.sigma = sigma;
  params_acg.termination_type = 'd_aicg';
         
  % ======================================
  % --------------- D-AICG ---------------
  % ======================================
  
  % Main loop
  while (o_iter < o_lmt)
    
    % If time is up, pre-maturely exit
    if (toc(t_start) > params.time_limit)
      history.runtime = toc(t_start);
      return;
    end
    
    % Set the outer iteration's acceleration parameters.
    a0 = (1 + sqrt(1 + 4 * A0)) / 2;
    A = A0 + a0;
    Z_xTilde = (A0 * Z_y0 + a0 * Z_x0) / A;
    
    % Compute the factorization of the pertrubed point:
    % X0 = Z_xTilde - lam * grad_f1_s(Z_xTilde)
    spo_at_Z_xTilde = spectral_oracle(Z_xTilde, []);
    Z_xTilde_grad_step = Z_xTilde - lambda * spo_at_Z_xTilde.grad_f1_s();
    [P, dg_sng, Q] = decomp_fn(Z_xTilde_grad_step);
    z_xTilde_grad_step = diag(dg_sng);
    
    % Set functions
    oracle_acg = ...
      @(sigma) vectorize_oracle(...
        sigma, z_xTilde_grad_step, lambda, spectral_oracle);
    
    % Set simple ACG params
    params_acg.x0 = diag(P' * Z_xTilde * Q);
    params_acg.X0_mat = Z_xTilde;
    params_acg.lambda = lambda;
    params_acg.mu = mu_fn(lambda);
    params_acg.L = L_prox_fn(lambda, M2 + xi); % upper curvature
    params_acg.L_est = L_prox_fn(lambda, M2 + xi); % estimated upper curvature
    params_acg.L_grad_f_s_est = L_prox_fn(lambda, M2 + xi);
    params_acg.t_start = t_start;
        
    % Call the ACG
    [model_acg, history_acg] = ACG(oracle_acg, params_acg);
    
    % Record output
    history.acg_iteration_values = ...
      [history.acg_iteration_values, history_acg.iter];
    history.iter = history.iter + history_acg.iter;
    
    % If time is up, pre-maturely exit
    if (toc(t_start) > params.time_limit)
      history.runtime = toc(t_start);
      return;
    end
    
    % Check for failure of the ACG method
    if (model_acg.status < 0)
      warning(['ACG failed! Status = ', num2str(model_acg.status)]);
      xi = xi * 2;
      spectral_oracle = ...
        @(x, sigma) redistribute_curvature(...
          x, sigma, o_spectral_oracle, xi);
      spo_at_Z_y0 = spectral_oracle(Z_y0, []);
      continue;
    end
       
    % Parse its output
    z_vec = model_acg.y;
    v_vec = model_acg.u;
    Z_y = P * spdiags(z_vec, 0, zR, zR) * Q';
    V_y = P * spdiags(v_vec, 0, zR, zR) * Q';
    
   % Check for failure of the descent inequality
   % i.e., check Delta(Z_y0; Z_y, v) <= epsilon.
    spo_at_Z_y = spectral_oracle(Z_y, z_vec);
    [Delta_at_Z_y0, prox_at_Z_y0, prox_at_Z_xTilde] = ...
      Delta_mu_fn(...
      1, spo_at_Z_y0, spo_at_Z_y, spo_at_Z_xTilde, ...
      Z_y0, Z_y, Z_xTilde, V_y);
    prox_base = ...
      max([abs(prox_at_Z_y0), abs(prox_at_Z_xTilde), macheps_Delta]);
    rel_err = (Delta_at_Z_y0 - model_acg.eta) / prox_base;
    if (rel_err >= macheps_Delta)
      warning(['Descent condition failed with lhs = ', ...
                num2str(Delta_at_Z_y0), ', and rhs = ', ...
                num2str(model_acg.eta), ', and base = ', ...
                num2str(prox_base)]);
      xi = xi * 2;
      % Update oracles
      spectral_oracle = ...
        @(x, sigma) redistribute_curvature(...
          x, sigma, o_spectral_oracle, xi);
      spo_at_Z_y0 = spectral_oracle(Z_y0, []);
      continue;
    end     
    
    % Call the refinement
    params.P = P;
    params.Q = Q;
    params.spo_Z0 = spo_at_Z_xTilde;
    refine_fn = ...
      @(Z, V) refine_icg(...
        params, spectral_oracle, M2 + xi, lambda, ...
        Z_xTilde, z_xTilde_grad_step, z_vec, v_vec);
    model_refine = refine_fn(Z_y, V_y);
    
    % Record some of the refinement values.
    orig_spo_z_hat = o_spectral_oracle(...
      model_refine.z_hat, model_refine.spo_at_z_hat.sigma);
    phi_at_z_hat = ...
      orig_spo_z_hat.f1_s() + orig_spo_z_hat.f2_s() + orig_spo_z_hat.f_n();
    cur_time = toc(t_start);
    history.iteration_values = ...
      [history.iteration_values, history.iter];
    history.function_values = ...
      [history.function_values, phi_at_z_hat];
    history.outer_function_values = ...
      [history.outer_function_values, phi_at_z_hat];
    history.time_values = ...
      [history.time_values, cur_time];
    history.outer_time_values = ...
      [history.outer_time_values, cur_time];
    
    % Check failure of the refinement
    % i.e., check Delta(Z_y_hat; Z_y, v) <= epsilon.
    Z_yHat = model_refine.z_hat;
    spo_at_Z_yHat = model_refine.spo_at_z_hat;
    [Delta_at_Z_yHat, prox_at_Z_yHat, prox_at_Z_xTilde] = ...
      Delta_mu_fn(...
        1, spo_at_Z_yHat, spo_at_Z_y, spo_at_Z_xTilde, ...
        Z_yHat, Z_y, Z_xTilde, V_y);
    prox_base = ...
      max([abs(prox_at_Z_yHat), abs(prox_at_Z_xTilde), macheps_Delta]);
    rel_err = (Delta_at_Z_yHat - model_acg.eta) / prox_base;
    if (rel_err >= macheps_Delta)
      warning(['Refinement condition failed with lhs = ', ...
                num2str(Delta_at_Z_yHat), ', and rhs = ', ...
                num2str(model_acg.eta), ', and base = ', ...
                num2str(prox_base)]);
      xi = xi * 2;
      spectral_oracle = ...
        @(x, sigma) redistribute_curvature(...
          x, sigma, o_spectral_oracle, xi);
      spo_at_Z_y0 = spectral_oracle(Z_y0, []);
      continue;
    end
    
    % Check termination based on the refined point
    model.x = model_refine.z_hat;
    model.v = model_refine.v_hat;
    if(norm_fn(model.v) <= tol)
      history.runtime = toc(t_start);
      return;
    end
    
    % Update Z_x
    Z_xNext = Z_x0 - a0 * (V_y + Z_xTilde - Z_y);
    Z_x = Omega_projection(Z_xNext);
    
    % Do nothing for the defaults (placeholder).
    if strcmp(steptype, 'constant')
      
    % Adjust lambda using the refinement residuals
    elseif strcmp(steptype, 'adaptive_v1')
      v1_hat = model_refine.q_hat - ...
        (lambda * (max([M2, 0]) + ...
         2 * max([m2, 0])) + 1) * (Z_y - model_refine.z_hat);
      norm_v1_hat = norm_fn(v1_hat);
      norm_v2_hat = norm_fn(model_refine.v_hat - v1_hat);
      ratio_v1_div_v2 = norm_v1_hat / norm_v2_hat;
      o_lambda = lambda;
      if (ratio_v1_div_v2 < 0.5)
        lambda = o_lambda * sqrt(0.5);
      elseif (ratio_v1_div_v2 > 2.0)
        lambda = o_lambda * sqrt(2);
      end
    end
                                     
    % Update D-AICG params for the next iteration
    if (is_monotone)
      % Monotonicty update (choose the 'best' y from {y, y^a})
      spo_at_Z_y_f1_only =  spectral_oracle(Z_y, z_vec);
      phi_at_Z_y = ...
        spo_at_Z_y_f1_only.f1_s() + spo_at_Z_y.f2_s() + spo_at_Z_y.f_n();
      if (phi_at_Z_y < phi_at_Z_y0)
        Z_y0 = Z_y;
        spo_at_Z_y0 = spo_at_Z_y;
      end
    else
      % Standard AG update (choose y^a)
      Z_y0 = Z_y;
      spo_at_Z_y0 = spo_at_Z_y;
    end
    Z_x0 = Z_x;
    A0 = A;
    o_iter = o_iter + 1;
                   
  end % End Main loop
  
end % function end