%{

FILE DATA
---------
Last Modified: 
  August 22, 2020
Coders: 
  Weiwei Kong

%}  

function [model, history] = IA_ICG(spectral_oracle, params)
% The accelerated inexact composite gradient (AICG) method.
% The inner-accelerated inexact composite gradient (IA-ICG) method.
%
% See Also:
%
%   **src.solvers.DA_ICG**
% 
% Note:
% 
%   Based on the paper:
%
%     Kong, W., & Monteiro, R. D. (2020). Accelerated Inexact Composite 
%     Gradient Methods for Nonconvex Spectral Optimization Problems. 
%     *arXiv preprint arXiv:2007.11772*.
%
% Arguments:
% 
%   spectral_oracle (SpectralOracle): The spectral oracle underlying the 
%     optimization problem.
% 
%   params.xi (double): Controls the amount of curvature that is initially
%     redistributed from $f_1$ to $f_2$. Defaults to ``params.M1``.
%
%   params.lambda (double): The initial stepsize $\lambda$. Defaults to 
%     ``5 / params.M1``.
%
%   params.steptype (character vector): Either ``'adaptive'`` or 
%     ``'constant'``. If it is ``adaptive``, then the stepsize is chosen
%     adaptively. If it is constant, then the stepsize is constant. Defaults
%     to ``'adaptive'``.
%
%   params.sigma (double): Controls the accuracy of the inner subroutine.
%     Defaults to ``(9 / 10 - max([params.lambda * (params.M1 - 
%     params.xi), 0]))``.
%
%   params.acg_steptype (character vector): Is either "variable" or 
%     "constant". If it is "variable", then the ACG call employs a line search 
%     subroutine to look for the appropriate upper curvature, with a starting 
%     estimate of $L_0 = \lambda (M / 100) + 1$. If "constant", then no 
%     subroutine is employed and the upper curvature remains fixed at 
%     $L_0 = \lambda M + 1$. Defaults to ``'variable'``. 
% 
% Returns: 
%   
%   A pair of structs containing model and history related outputs of the 
%   solved problem associated with the oracle and input parameters.
%

  % Set global constants
  DELTA_TOL = 1e-2;

  % Timer start.
  t_start = tic;

  % -----------------------------------------------------------------------
  %% PRE-PROCESSING
  % -----------------------------------------------------------------------
  
  % Set REQUIRED input params.
  Z0 = params.x0; 
  m2 = params.m2;
  M1 = params.M1;
  M2 = params.M2;
  norm_fn = params.norm_fn;
  opt_tol = params.opt_tol;
  decomp_fn = params.decomp_fn;
  
  % Fill in OPTIONAL input params.
  params = set_default_params(params);
  
  % Set other input params.
  xi = params.xi;
  lambda = params.lambda;
  steptype = params.steptype;
  
  % Solver params.
  time_limit = params.time_limit;
  iter_limit = params.iter_limit;
  
  % History params.
  if params.i_logging
    spectral_oracle.eval(Z0);
    function_values = spectral_oracle.f_s() + spectral_oracle.f_n();
    iteration_values = 0;
    time_values = 0;
  end
  
  % Initialize the model struct in case there is an ACG failure.
  x = Z0;
  v = -Inf;
     
  % Initialize some auxillary functions and constants.
  iter = 0; % Set to zero to offset ACG iterations
  outer_iter = 1;
  mu_fn = @(lam, m) 1 / 2;
  L_prox_fn = @(lam, L) lam * L + 1; % prox curvature function
  params_acg = params;
  params_acg.termination_type = 'aicg';
  [zM, zN] = size(Z0);
  zR = min(zM, zN);  
  
  % Convexify the oracle.
  o_spectral_oracle = copy(spectral_oracle);
  spectral_oracle.redistribute_curvature(xi);
       
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
    %% ACG CALL AND POST-PROCESSING
    % ---------------------------------------------------------------------    
    
    % Compute the factorization of the perturbed point:
    %   X0 = Z0 - lam * grad_f1_s(Z0)
    spo_at_Z0 = copy(spectral_oracle.eval(Z0));
    Z0_grad_step = Z0 - lambda * spo_at_Z0.grad_f1_s();
    [P, dg_sng, Q] = decomp_fn(Z0_grad_step);
    z0_grad_step = diag(dg_sng);
    
    % Set functions
    oracle_acg = copy(spectral_oracle);
    oracle_acg.vector_linear_proxify(lambda, z0_grad_step);
    
    % Set simple ACG params
    params_acg.x0 = diag(P' * Z0 * Q);
    params_acg.X0_mat = Z0;
    params_acg.lambda = lambda;
    params_acg.mu = mu_fn(lambda);
    params_acg.L = L_prox_fn(lambda, M2 + xi); % upper curvature
    params_acg.L_est = L_prox_fn(lambda, M2 + xi); % est. upper curvature
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
    
    % Check for failure of the ACG method
    if (model_acg.status < 0)
      warning(['ACG failed! Status = ', num2str(model_acg.status)]);
      xi = xi * 2;
      spectral_oracle = copy(o_spectral_oracle);
      spectral_oracle.redistribute_curvature(xi);
      continue;
    elseif (model_acg.status == 0)
      continue;
    end
    
    % Parse its output
    z_vec = model_acg.y;
    v_vec = model_acg.u;
    Z = P * spdiags(z_vec, 0, zR, zR) * Q';
    V = P * spdiags(v_vec, 0, zR, zR) * Q';
    
    % ---------------------------------------------------------------------
    %% OTHER UPDATES
    % ---------------------------------------------------------------------
    
    % Check for failure of the (relative) descent inequality
    % i.e. check Delta(y0; y, v) <= epsilon.
    spo_at_Z = copy(spectral_oracle.spectral_eval(Z, z_vec));
    [Delta_at_Z0, prox_at_Z, prox_at_Z0] = Delta_mu_fn(...
      params, 1, spo_at_Z0, spo_at_Z, spo_at_Z0, Z0, Z, Z0, V);
    prox_base = ...
      max([abs(prox_at_Z), abs(prox_at_Z0), DELTA_TOL]);
    rel_err = (Delta_at_Z0 - model_acg.eta) / prox_base;
    
    % If a failure occurs, redistribute the curvature.
    if (rel_err >= DELTA_TOL)
      warning(['Descent condition failed with lhs = ', ...
                num2str(Delta_at_Z0), ', and rhs = ', ...
                num2str(model_acg.eta), ', and base = ', ...
                num2str(prox_base)]);
      xi = xi * 2;
      spectral_oracle = copy(o_spectral_oracle);
      spectral_oracle.redistribute_curvature(xi);
      continue;
    end
    
    % Call the refinement
    params.P = P;
    params.Q = Q;
    params.spo_Z0 = spo_at_Z0;
    model_refine = refine_ICG(...
        spectral_oracle, params, M2 + xi, lambda, Z0, z0_grad_step, ...
        z_vec, v_vec);
    
    % Check failure of the refinement, 
    % i.e., check Delta(y_hat; y, v) <= epsilon.
    Z_hat = model_refine.z_hat;
    spo_at_Z_hat = model_refine.spo_at_z_hat;
    [Delta_at_Z_hat, prox_at_Z_hat, prox_at_Z0] = ...
      Delta_mu_fn(...
        params, 1, spo_at_Z_hat, spo_at_Z, spo_at_Z0, Z_hat, Z, Z0, V);
    prox_base = ...
      max([abs(prox_at_Z_hat), abs(prox_at_Z0), DELTA_TOL]);
    rel_err = (Delta_at_Z_hat - model_acg.eta) / prox_base;
    
    % If a failure occurs, redistribute the curvature.
    if (rel_err >= DELTA_TOL)
      warning(['Refinement condition failed with lhs = ', ...
                num2str(Delta_at_Z_hat), ', and rhs = ', ...
                num2str(model_acg.eta), ', and base = ', ...
                num2str(prox_base)]);
      xi = xi * 2;
      spectral_oracle = copy(o_spectral_oracle);
      spectral_oracle.redistribute_curvature(xi);
      continue;
    end
    
    % Check termination based on the refined point
    x = model_refine.z_hat;
    v = model_refine.v_hat;
    if(norm_fn(v) <= opt_tol)
      break;
    end
    
    % Do nothing for the defaults (placeholder).
    if strcmp(steptype, 'constant')
      
    % Adjust lambda using the refinement residuals
    elseif strcmp(steptype, 'adaptive')
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
    
    % Update history.
    if params.i_logging
      o_spectral_oracle.eval(x);
      function_values(end + 1) = ...
        o_spectral_oracle.f_s() + o_spectral_oracle.f_n();
      iteration_values(end + 1) = iter;
      time_values(end + 1) = toc(t_start);
    end
                              
    % Update AICG params.
    Z0 = Z;
    outer_iter = outer_iter + 1;
                   
  end % End Main loop
  
  % -----------------------------------------------------------------------
  %% FINAL POST-PROCESSING
  % -----------------------------------------------------------------------
  
  % Prepare the model and history
  model.x = x;
  model.v = v;
  model.last_lambda = lambda;
  history.iter = iter;
  history.outer_iter = outer_iter;
  history.runtime = toc(t_start);
  if params.i_logging
    history.function_values = function_values;
    history.iteration_values = iteration_values;
    history.time_values = time_values;
  end
  
end % function end

% ======================================================
% --------------- Other Helper Functions ---------------  
% ======================================================

% Fills in parameters that were not set as input
function params = set_default_params(params)

  % xi = M1
  if (~isfield(params, 'xi'))
    params.xi = params.M1;
  end
  
  % lambda = 5 / M1.
  if (params.M1 - params.xi <= 0)
    if ~isfield(params, 'lambda')
      params.lambda = 5 / params.M1;
    end
  else
    if ~isfield(params, 'lambda')
      params.lambda = 1 / (4 * params.M1);
    end
  end
  
  % steptype = 'adaptive'
  if ~isfield(params, 'steptype')
    params.steptype = 'adaptive';
  end
  
  % sigma = sqrt(9 / 10 - max([lambda * (M1 - xi), 0])).
  if ~isfield(params, 'sigma')
    incumb_sqr_sigma = ...
      (9 / 10 - max([params.lambda * (params.M1 - params.xi), 0]));
    if (incumb_sqr_sigma < 0)
      error('lambda is too large to set sigma!');
    end
    params.sigma = sqrt(incumb_sqr_sigma);
  end
  
  % acg_steptype = 'variable'
  if ~isfield(params, 'acg_steptype')
    params.acg_steptype = 'variable';
  end
  
  % i_logging = false
  if (~isfield(params, 'i_logging')) 
    params.i_logging = false;
  end

end