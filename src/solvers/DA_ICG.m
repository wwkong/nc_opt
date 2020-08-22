%{

FILE DATA
---------
Last Modified: 
  August 22, 2020
Coders: 
  Weiwei Kong

%}  

function [model, history] = DA_ICG(spectral_oracle, params)
% The doubly-accelerated inexact composite gradient (DA-ICG) method.
%
% See Also:
%
%   **src.solvers.IA_ICG**
% 
% Note:
% 
%   Based on the paper:
%
%     **[1]** Kong, W., & Monteiro, R. D. (2020). Accelerated Inexact 
%     Composite Gradient Methods for Nonconvex Spectral Optimization Problems. 
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
%     Defaults to ``(1 / 2 - max([params.lambda * (params.M1 - 
%     params.xi), 0]))``.
%
%   params.acg_steptype (character vector): Is either "variable" or 
%     "constant". If it is "variable", then the ACG call employs a line search 
%     subroutine to look for the appropriate upper curvature, with a starting 
%     estimate of $L_0 = \lambda (M / 100) + 1$. If "constant", then no 
%     subroutine is employed and the upper curvature remains fixed at 
%     $L_0 = \lambda M + 1$. Defaults to ``'variable'``. 
%
%   params.Omega_projection (function handle): A one argument function, which
%     when evaluated at $x$, computes ${\rm Proj}_\Omega(x)$. For details
%     see the definition of $\Omega$ in **[1]**. Defaults to ``@(X) X``.
%
%   params.is_monotone (bool): If ``true``, then the sequence of outer 
%     iterates forms a monotonically nonincreasing sequence with respect to 
%     the objective function. If ``false``, then no such property is
%     guaranteed. Defaults to ``true``.
%
% Returns: 
%   
%   A pair of structs containing model and history related outputs of the 
%   solved problem associated with the oracle and input parameters.
%
  
  % Set global constants
  DELTA_TOL = 1e-2;

  % Timer start
  t_start = tic;
                            
  % -----------------------------------------------------------------------
  %% PRE-PROCESSING
  % -----------------------------------------------------------------------
  
  % Main params
  Z_y0 = params.x0; % This is the initial point of the algorithm
  m2 = params.m2;
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
  Omega_projection = params.Omega_projection;
  is_monotone = params.is_monotone;
  
  % History params.
  if params.i_logging
    spectral_oracle.eval(Z_y0);
    function_values = spectral_oracle.f_s() + spectral_oracle.f_n();
    iteration_values = 0;
    time_values = 0;
  end
  
  % Solver params.
  time_limit = params.time_limit;
  iter_limit = params.iter_limit;
    
  % Initialize the model struct in case there is an ACG failure.
  x = Z_y0;
  v = -Inf;
  
  % Initialize some auxillary functions and constants.
  spo_at_Z_y0 = copy(spectral_oracle.eval(Z_y0));
  phi_at_Z_y0 = spo_at_Z_y0.f_s() + spo_at_Z_y0.f_n();
  iter = 0; % Set to zero to offset ACG iterations
  outer_iter = 1;
  mu_fn = @(lam) 1 / 2;
  L_prox_fn = @(lam, L) lam * L + 1; % prox curvature function
  params_acg = params;
  params_acg.termination_type = 'd_aicg';
  [zM, zN] = size(Z_y0);
  zR = min(zM, zN);
  
  % Set outer acceleration parameters.
  A0 = 0;
  Z_x0 = Z_y0;
  
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
    
    % Set the outer iteration's acceleration parameters.
    a0 = (1 + sqrt(1 + 4 * A0)) / 2;
    A = A0 + a0;
    Z_xTilde = (A0 * Z_y0 + a0 * Z_x0) / A;
    
    % ---------------------------------------------------------------------
    %% ACG CALL AND POST-PROCESSING
    % ---------------------------------------------------------------------    
    
    % Compute the factorization of the pertrubed point:
    %   X0 = Z_xTilde - lam * grad_f1_s(Z_xTilde)
    spo_at_Z_xTilde = copy(spectral_oracle.eval(Z_xTilde));
    Z_xTilde_grad_step = Z_xTilde - lambda * spo_at_Z_xTilde.grad_f1_s();
    [P, dg_sng, Q] = decomp_fn(Z_xTilde_grad_step);
    z_xTilde_grad_step = diag(dg_sng);
    
    % Set functions
    oracle_acg = copy(spectral_oracle);
    oracle_acg.vector_linear_proxify(lambda, z_xTilde_grad_step);
    
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
    iter = iter + history_acg.iter;
    
    % Check for failure of the ACG method
    if (model_acg.status < 0)
      warning(['ACG failed! Status = ', num2str(model_acg.status)]);
      xi = xi * 2;
      spectral_oracle = copy(o_spectral_oracle);
      spectral_oracle.redistribute_curvature(xi);
      spo_at_Z_y0 = copy(spectral_oracle.eval(Z_y0));
      continue;
    elseif (model_acg.status == 0)
      continue;
    end
       
    % Parse its output
    z_vec = model_acg.y;
    v_vec = model_acg.u;    
    Z_y = P * spdiags(z_vec, 0, zR, zR) * Q';
    V_y = P * spdiags(v_vec, 0, zR, zR) * Q';
 
    % ---------------------------------------------------------------------
    %% OTHER UPDATES
    % ---------------------------------------------------------------------
    
    % Check for failure of the descent inequality
    % i.e., check Delta(Z_y0; Z_y, v) <= epsilon.
    spo_at_Z_y = copy(spectral_oracle.spectral_eval(Z_y, z_vec));   
    [Delta_at_Z_y0, prox_at_Z_y0, prox_at_Z_xTilde] = Delta_mu_fn(...
      params, 1, spo_at_Z_y0, spo_at_Z_y, spo_at_Z_xTilde, ...
      Z_y0, Z_y, Z_xTilde, V_y);    
    prox_base = ...
      max([abs(prox_at_Z_y0), abs(prox_at_Z_xTilde), DELTA_TOL]);
    rel_err = (Delta_at_Z_y0 - model_acg.eta) / prox_base;
    if (rel_err >= DELTA_TOL)
      warning(['Descent condition failed with lhs = ', ...
                num2str(Delta_at_Z_y0), ', and rhs = ', ...
                num2str(model_acg.eta), ', and base = ', ...
                num2str(prox_base)]);
      xi = xi * 2;
      spectral_oracle = copy(o_spectral_oracle);
      spectral_oracle.redistribute_curvature(xi); 
      spo_at_Z_y0 = copy(spectral_oracle.eval(Z_y0));
      continue;
    end     
    
    % Call the refinement
    params.P = P;
    params.Q = Q;
    params.spo_Z0 = copy(spo_at_Z_xTilde);
    model_refine = refine_ICG(...
        spectral_oracle, params, M2 + xi, lambda, Z_xTilde, ...
        z_xTilde_grad_step, z_vec, v_vec);
    
    % Check failure of the refinement
    % i.e., check Delta(Z_y_hat; Z_y, v) <= epsilon.
    Z_yHat = model_refine.z_hat;
    spo_at_Z_yHat = copy(model_refine.spo_at_z_hat);
    [Delta_at_Z_yHat, prox_at_Z_yHat, prox_at_Z_xTilde] = ...
      Delta_mu_fn(...
        params, 1, spo_at_Z_yHat, spo_at_Z_y, spo_at_Z_xTilde, ...
        Z_yHat, Z_y, Z_xTilde, V_y);
    prox_base = ...
      max([abs(prox_at_Z_yHat), abs(prox_at_Z_xTilde), DELTA_TOL]);
    rel_err = (Delta_at_Z_yHat - model_acg.eta) / prox_base;
    if (rel_err >= DELTA_TOL)
      warning(['Refinement condition failed with lhs = ', ...
                num2str(Delta_at_Z_yHat), ', and rhs = ', ...
                num2str(model_acg.eta), ', and base = ', ...
                num2str(prox_base)]);
      xi = xi * 2;
      spectral_oracle = copy(o_spectral_oracle);
      spectral_oracle.redistribute_curvature(xi); 
      spo_at_Z_y0 = copy(spectral_oracle.eval(Z_y0));
      continue;
    end
    
    % Check termination based on the refined point
    x = model_refine.z_hat;
    v = model_refine.v_hat;
    if(norm_fn(v) <= opt_tol)
      break;
    end
    
    % Update Z_x
    Z_xNext = Z_x0 - a0 * (V_y + Z_xTilde - Z_y);
    Z_x = Omega_projection(Z_xNext);
    
    % Do nothing for the defaults (placeholder).
    if strcmp(steptype, 'constant')
      
    % Adjust lambda using the refinement residuals
    elseif strcmp(steptype, 'adaptive')
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
    
    % Update history.
    if params.i_logging
      o_spectral_oracle.eval(x);
      function_values(end + 1) = ...
        o_spectral_oracle.f_s() + o_spectral_oracle.f_n();
      iteration_values(end + 1) = iter;
      time_values(end + 1) = toc(t_start);
    end
                                     
    % Update D-AICG params for the next iteration
    if (is_monotone)
      % Monotonicty update (choose the 'best' y from {y, y^a})
      spo_at_Z_y_f1_only = copy(spectral_oracle.spectral_eval(Z_y, z_vec));
      phi_at_Z_y = ...
        spo_at_Z_y_f1_only.f1_s() + spo_at_Z_y.f2_s() + spo_at_Z_y.f_n();
      if (phi_at_Z_y < phi_at_Z_y0)
        Z_y0 = Z_y;
        spo_at_Z_y0 = copy(spo_at_Z_y);
      end
    else
      % Standard AG update (choose y^a)
      Z_y0 = Z_y;
      spo_at_Z_y0 = copy(spo_at_Z_y);
    end
    phi_at_Z_y0 = phi_at_Z_y;
    Z_x0 = Z_x;
    A0 = A;
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
      (1 / 2 - max([params.lambda * (params.M1 - params.xi), 0]));
    if (incumb_sqr_sigma < 0)
      error('lambda is too large to set sigma!');
    end
    params.sigma = sqrt(incumb_sqr_sigma);
  end
  
  % acg_steptype = 'variable'
  if ~isfield(params, 'acg_steptype')
    params.acg_steptype = 'variable';
  end
  
  % Omega_projection = @(X) X
  if ~isfield(params, 'Omega_projection')
    params.Omega_projection = @(X) X;
  end
  
  % is_monotone = true
  if ~isfield(params, 'is_monotone')
    params.is_monotone = true;
  end 
  
  % i_logging = false
  if (~isfield(params, 'i_logging')) 
    params.i_logging = false;
  end

end