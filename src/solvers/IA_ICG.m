%{

The accelerated inexact composite gradient (AICG) method.

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

function [model, history] = IA_ICG(spectral_oracle, params)
  
  % Timer start
  t_start = tic;
                            
  % ===========================================
  % --------------- Parse Input ---------------
  % ===========================================
   
  % Induced norm
  norm_fn = params.norm_fn;
  iter_limit = params.iter_limit;
  time_limit = params.time_timit;
  
  % Main params
  Z0 = params.x0; % This is the initial point of the algorithm
  m2 = params.m2;
  M1 = params.M1;
  M2 = params.M2;
  tol = params.tol;
  decomp_fn = params.decomp_fn;
  macheps_Delta = 1e-2;
  
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
  
  % Initialize
  [zM, zN] = size(Z0);
  zR = min(zM, zN);
  model.x = Z0;
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
  if ((9 / 10 - max([lambda * (M1 - xi), 0])) < 0)
    error('lambda is too large to set sigma!');
  end
  sigma = sqrt(9 / 10 - max([lambda * (M1 - xi), 0]));
   
  % Curvature functions
  mu_fn = @(lam, m) 1 / 2;
  L_prox_fn = @(lam, L) lam * L + 1; % prox curvature function

  % Params for the first ACG method call (and possibly subsequent calls)
  params_acg = params;
  params_acg.sigma = sigma;
  params_acg.termination_type = 'aicg';
       
  % ====================================
  % --------------- AICG ---------------
  % ====================================
  
  % Main loop
  while (o_iter < o_lmt)
    
    % If time is up, pre-maturely exit
    if (toc(t_start) > params.time_limit)
      history.runtime = toc(t_start);
      return;
    end
    
    % Compute the factorization of the perturbed point:
    % X0 = Z0 - lam * grad_f1_s(Z0)
    spo_at_Z0 = spectral_oracle(Z0, []);
    Z0_grad_step = Z0 - lambda * spo_at_Z0.grad_f1_s();
    [P, dg_sng, Q] = decomp_fn(Z0_grad_step);
    z0_grad_step = diag(dg_sng);
    
    % Set functions
    oracle_acg = ...
      @(sigma) vectorize_oracle(...
        sigma, z0_grad_step, lambda, spectral_oracle);
    
    % Set simple ACG params
    params_acg.x0 = diag(P' * Z0 * Q);
    params_acg.X0_mat = Z0;
    params_acg.lambda = lambda;
    params_acg.mu = mu_fn(lambda);
    params_acg.L = L_prox_fn(lambda, M2 + xi); % upper curvature
    params_acg.L_est = L_prox_fn(lambda, M2 + xi); % estimated upper curvature
    params_acg.L_grad_f_s_est = L_prox_fn(lambda, M2 + xi);
    params_acg.t_start = t_start;
    
    % Set other ACG params
    phi_at_Z0 = spo_at_Z0.f1_s() + spo_at_Z0.f2_s() + spo_at_Z0.f_n();
    params_acg.f1_at_Z0 = spo_at_Z0.f1_s();
    params_acg.Dg_Pt_grad_f1_at_Z0_Q = ...
      diag(P' * spo_at_Z0.grad_f1_s() * Q);
    params_acg.phi_at_Z0 = phi_at_Z0;
    params_acg.aicg_M1 = M1 - xi;
        
    % Call the ACG
    [model_acg, history_acg] = ACG(oracle_acg, params_acg);
    
    % Record output
    iter = iter + history_acg.iter;
    
    % If time is up, pre-maturely exit
    if (toc(t_start) > time_limit)
      history.runtime = toc(t_start);
      break;
    end
    
    % Check for failure of the ACG method
    if (model_acg.status < 0)
      warning(['ACG failed! Status = ', num2str(model_acg.status)]);
      xi = xi * 2;
      spectral_oracle = ...
        @(x, sigma) ...
           redistribute_curvature(x, sigma, o_spectral_oracle, xi);
      continue;
    end
    
    % Parse its output
    z_vec = model_acg.y;
    v_vec = model_acg.u;
    Z = P * spdiags(z_vec, 0, zR, zR) * Q';
    V = P * spdiags(v_vec, 0, zR, zR) * Q';
    
    % Check for failure of the (relative) descent inequality
    % i.e. check Delta(y0; y, v) <= epsilon.
    spo_at_Z = spectral_oracle(Z, z_vec);
    [Delta_at_Z0, prox_at_Z, prox_at_Z0] = ...
      Delta_mu_fn(1, spo_at_Z0, spo_at_Z, spo_at_Z0, Z0, Z, Z0, V);
    prox_base = ...
      max([abs(prox_at_Z), abs(prox_at_Z0), macheps_Delta]);
    rel_err = (Delta_at_Z0 - model_acg.eta) / prox_base;
    
    % If a failure occurs, redistribute the curvature.
    if (rel_err >= macheps_Delta)
      warning(['Descent condition failed with lhs = ', ...
                num2str(Delta_at_Z0), ', and rhs = ', ...
                num2str(model_acg.eta), ', and base = ', ...
                num2str(prox_base)]);
      xi = xi * 2;
      % Update oracles
      spectral_oracle = ...
        @(x, sigma) redistribute_curvature(...
          x, sigma, o_spectral_oracle, xi);
      continue;
    end
    
    % Call the refinement
    params.P = P;
    params.Q = Q;
    params.spo_Z0 = spo_at_Z0;
    refine_fn = ...
      @(Z, V) refine_icg(...
        params, spectral_oracle, M2 + xi, lambda, ...
        Z0, z0_grad_step, z_vec, v_vec);
    model_refine = refine_fn(Z, V);
    
    % Check failure of the refinement, 
    % i.e., check Delta(y_hat; y, v) <= epsilon.
    Z_hat = model_refine.z_hat;
    spo_at_Z_hat = model_refine.spo_at_z_hat;
    [Delta_at_Z_hat, prox_at_Z_hat, prox_at_Z0] = ...
      Delta_mu_fn(1, spo_at_Z_hat, spo_at_Z, spo_at_Z0, Z_hat, Z, Z0, V);
    prox_base = ...
      max([abs(prox_at_Z_hat), abs(prox_at_Z0), macheps_Delta]);
    rel_err = (Delta_at_Z_hat - model_acg.eta) / prox_base;
    
    % If a failure occurs, redistribute the curvature.
    if (rel_err >= macheps_Delta)
      warning(['Refinement condition failed with lhs = ', ...
                num2str(Delta_at_Z_hat), ', and rhs = ', ...
                num2str(model_acg.eta), ', and base = ', ...
                num2str(prox_base)]);
      xi = xi * 2;
      % Update oracles
      spectral_oracle = ...
        @(x, sigma) redistribute_curvature(...
          x, sigma, o_spectral_oracle, xi);
      continue;
    end
    
    % Check termination based on the refined point
    model.x = model_refine.z_hat;
    model.v = model_refine.v_hat;
    if(norm_fn(model.v) <= tol)
      history.runtime = toc(t_start);
      return;
    end
    
    % Do nothing for the defaults (placeholder).
    if strcmp(steptype, 'constant')
      
    % Adjust lambda using the refinement residuals
    elseif strcmp(steptype, 'adaptive_v1')
      v1_hat = model_refine.q_hat - ...
        (lambda * (max([M2, 0]) + 2 * max([m2, 0])) + 1) * ...
        (Z - model_refine.z_hat);
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
                              
    % Update AICG params.
    Z0 = Z;
                   
  end % End Main loop
  
end % function end