%{

FILE DATA
---------
Last Modified: 
  August 2, 2020
Coders: 
  Weiwei Kong

%}

function [model, history] = AIPP(oracle, params)
% Variants of the accelerated inexact proximal point (AIPP) method
% 
% See Also:
%   
%   **src.solvers.ACG**
% 
% Note:
% 
%   Based on the papers:
%
%   **[1]** Kong, W., Melo, J. G., & Monteiro, R. D. (2019). Complexity of a
%   quadratic penalty accelerated inexact proximal point method for solving 
%   linearly constrained nonconvex composite programs. *SIAM Journal on 
%   Optimization, 29*\(4), 2566-2593.
%
%   **[2]** Kong, W., Melo, J. G., & Monteiro, R. D. (2020). An efficient 
%   adaptive accelerated inexact proximal point method for solving linearly 
%   constrained nonconvex composite problems. *Computational Optimization and 
%   Applications, 76*\(2), 305-346. 
%
%   **[3]** Kong, W., & Monteiro, R. D. (2019). An accelerated inexact 
%   proximal point method for solving nonconvex-concave min-max problems. 
%   *arXiv preprint arXiv:1905.13433*.
%
%   In particular, $\tau$ is updated using the rule in [3].
%
% Arguments:
%
%   oracle (Oracle): The oracle underlying the optimization problem.
%
%   params.aipp_type (character vector): Specifies which AIPP variant to use.
%     More specifically, 'aipp' is the AIPP method from [1], while 'aipp_c', 
%     'aipp_v1', and 'aipp_v2' are the R-AIPPc, R-AIPPv1, and R-AIPPv2 methods
%     from [2]. Defaults to ``'aipp_v2'``.
%
%   params.sigma (double): Determines the accuracy of the inner ACG call (see 
%     $\sigma$ from [1]). Defaults to ``0.3``.
%
%   params.theta (double): Determines the accuracy of the inner R-ACG call 
%     (see $\theta$ from [2]). Defaults to ``4``.
%
%   params.acg_steptype (character vector): Is either "variable" or 
%     "constant". If it is "variable", then the ACG call employs a line search 
%     subroutine to look for the appropriate upper curvature, with a starting 
%     estimate of $L_0 = \lambda (M / 100) + 1$. If "constant", then no 
%     subroutine is employed and the upper curvature remains fixed at 
%     $L_0 = \lambda M + 1$. Defaults to ``'variable'``. 
%
%   params.acg_ls_multiplier (double): Determines how quickly the line search 
%     subroutine of the ACG call (multiplicatively) increases its estimate. 
%     Defaults to ``1.25``.
%
%   params.lambda (double): The initial stepsize (see $\lambda$ from [2]). 
%     Defaults to ``1 / m``.
%
%   params.mu_fn (function handle): Determines the strong convexity parameter 
%     $\mu$ given to the ACG call as a function of lambda, the stepsize.
%     Defaults to ``@(lambda) 1``.
%
%   params.tau_mult (double): An auxiliary parameter constant used to 
%     determine the first value of $\tau$. Defaults to ``10``.
%
%   params.tau (double): A parameter constant that determines the accuracy of 
%     the inner ACG call (see $\tau$ from [2, 3]). Defaults to ``tau_mult * 
%     (lambda M + 1)`` where ``lambda`` is the initial stepsize and 
%     ``tau_mult`` is the constant in ``params.tau_mult``.
%
% Returns:
%   
%   A pair of structs containing model and history related outputs of the 
%   solved problem associated with the oracle and input parameters.
%
  
  % Start the timer.
  t_start = tic;
  
  % -----------------------------------------------------------------------
  %% PRE-PROCESSING
  % -----------------------------------------------------------------------
  
  % Set REQUIRED input params.
  z0 = params.x0; 
  M = params.M;
  norm_fn = params.norm_fn;
  opt_tol = params.opt_tol; 
  
  % Fill in OPTIONAL input params.
  params = set_default_params(params);
  
  % Set other input params.
  lambda = params.lambda;
  mu_fn = params.mu_fn;
  m = params.m;
  
  % History params.
  if params.i_logging
    oracle.eval(z0);
    function_values = oracle.f_s() + oracle.f_n();
    iteration_values = 0;
    time_values = 0;
  end  
  
  % Solver params
  iter_limit = params.iter_limit;
  time_limit = params.time_limit;
  aipp_type = params.aipp_type;
  i_variable_tau = params.i_variable_tau;
  i_increase_lambda = params.i_increase_lambda;
  
  % Data checks
  if (strcmp(aipp_type, 'aipp_c') && params.lambda > 1 / m)
    error('Constant stepsize parameter lambda is too large!')
  end
  
  % Initialize the model struct in case the ACG fails
  x = z0;
  v = -Inf;
  
  % Initialize some auxillary functions and constants.
  iter = 0; % Set to zero to offset ACG iterations
  outer_iter = 1;
  M_est = M;
  L = max(m, M);
  L_prox_fn = @(lam, L) lam * L + 1;
  L_grad_f_s_est = L;
  params_acg = params;
  params_acg.mult_L = params.acg_ls_multiplier;

  % Initialize other input parameters split by aipp_type.
  if (any(strcmp(aipp_type, {'aipp_c', 'aipp_v1', 'aipp_v2'})))
    tau = params.tau;
    params_acg.termination_type = 'gd';
  elseif (strcmp(aipp_type, 'aipp'))
    params_acg.termination_type = 'aipp';
  else
    error(...
      ['Incorrect AIPP type specified! Valid types are: ', ...
       '{aipp, aipp_c, aipp_v1, aipp_v2}'])
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
    %% ACG CALL AND POST-PROCESSING
    % ---------------------------------------------------------------------
    
    % Set up the ACG oracle. 
    % NOTE: We need to explicitly call copy() because the 'Oracle' class
    % inherits from the 'handle' class
    oracle_acg = copy(oracle);
    oracle_acg.proxify(lambda, z0);
    
    % Set ACG parameters.
    params_acg.lambda = lambda;
    params_acg.x0 = z0;
    params_acg.z0 = z0;
    params_acg.mu = mu_fn(lambda);
    params_acg.L = L_prox_fn(lambda, M); % upper curvature
    params_acg.L_est = L_prox_fn(lambda, M_est); % est. upper curvature
    params_acg.L_grad_f_s_est = L_prox_fn(lambda, L_grad_f_s_est);
    params_acg.t_start = t_start;
        
    % Call the ACG.
    [model_acg, history_acg] = ACG(oracle_acg, params_acg);
    
    % Parse ACG outputs that are invariant of output status.
    iter = iter + history_acg.iter;
    M_est = (model_acg.L_est - 1) / lambda;
    
    % Check for failure of the ACG method.
    if (model_acg.status < 0)
      if (any(strcmp(aipp_type, {'aipp_v1', 'aipp_v2'})))
        i_increase_lambda = false;
        lambda = lambda / 2;
        continue;
      else
        error('Convexity assumption violated for R-AIPPc method!')
      end     
    elseif (model_acg.status == 0)
      continue;
    end
    
    % ---------------------------------------------------------------------
    %% OTHER UPDATES
    % ---------------------------------------------------------------------
    
    % Check for termination using the refinement
    model_refine = refine_IPP(...
      oracle, params, L, lambda, z0, model_acg.y, model_acg.u);
    x = model_refine.z_hat;
    v = model_refine.v_hat;
    if(norm_fn(v) <= opt_tol)
      break;
    end
    
    % Update tau if necessary.
    if (i_variable_tau)
      residual_ratio = ...
        model_refine.residual_v_hat / model_refine.residual_1;
      high_ratio = 1.5;
      low_ratio = 1.2;
      if (residual_ratio > high_ratio)
        tau = tau * high_ratio / residual_ratio;
      elseif (residual_ratio < low_ratio)
        tau = tau * low_ratio / residual_ratio;
      end
    end
       
    % Update lambda if we are using an adaptive stepsize variant.
    if (any(strcmp(aipp_type, {'aipp_v1', 'aipp_v2'})))
            
      % Restart if we have Delta > sigma / 2 * |v + z0 - z| ^ 2.
      if (params_acg.termination_type == "gd")
        if (2 * model_refine.Delta > tau / (lambda * M + 1) * ...
            norm_fn(model_acg.u + z0 - model_acg.y) ^ 2)
          i_increase_lambda = false;
          lambda = lambda / 2;
          continue;
        end
      end
      
      % If there is no failure and lambda is sufficiently small, then 
      % double lambda up to some point.
      lambda_upper = max([100 * 1 / m, 10 * L]); 
      if (i_increase_lambda)
        lambda = min(lambda_upper, lambda * 2);
      end 
    end
    
    % ---------------------------------------------------------------------
    %% ORIGINAL AIPP'S PHASE II CHECK
    % ---------------------------------------------------------------------
    if strcmp(aipp_type, 'aipp')
      if (norm_fn(model_acg.u + z0 - model_acg.y) <= lambda * opt_tol / 5)
        % Final ACG call.
        params_acg.termination_type = 'aipp_phase2';
        params_acg.epsilon_bar = opt_tol ^ 2 / (32 * (M + 2 * m));        
        [model_acg, history_acg] = ACG(oracle_acg, params_acg);
        iter = iter + history_acg.iter;
        % Final refinement.
        model_refine = refine_IPP(L, lambda, z0, model_acg.y, model_acg.u);
        x = model_refine.z_hat;
        v = model_refine.v_hat;
        break;
      end
    end
    
    % Update history
    if params.i_logging
      oracle.eval(model_acg.y);
      function_values(end + 1) = oracle.f_s() + oracle.f_n();
      iteration_values(end + 1) = iter;
      time_values(end + 1) = toc(t_start);
    end
    
    % Update iterates
    z0 = model_acg.y;
    outer_iter = outer_iter + 1;
                   
  end % main loop end
  
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

  % aipp_type = 'aipp_v2' ---> necessary to set other default params
  if (~isfield(params, 'aipp_type'))
    params.aipp_type = 'aipp_v2';
    aipp_type = params.aipp_type;
  else
    aipp_type = params.aipp_type;
  end
  
  % m = 1
  if(~isfield(params, 'm'))
    if (strcmp(aipp_type, 'aipp') || strcmp(aipp_type, 'aipp_c'))
      warning('No lower curvature m. Using m = M instead.');
    end
    params.m = params.M;
  end
  
  % sigma = 0.3
  if (~isfield(params, 'sigma'))
    params.sigma = 0.3;
  end
  
  % theta = 4
  if(~isfield(params, 'theta') && strcmp(aipp_type, 'aipp'))
    params.theta = 2 / (1 - params.sigma);
  elseif(~isfield(params, 'theta'))
    params.theta = 4;
  end
  
  % acg_ls_multiplier = 1.25
  if (~isfield(params, 'acg_ls_multiplier'))
    params.acg_ls_multiplier = 1.25;
  end
  
  % lambda ----------------------------------------> dependent on aipp_type
  if strcmp(aipp_type, 'aipp')
    params.lambda = 1 / (2 * params.m);
    params.i_increase_lambda = false;
  elseif strcmp(aipp_type, 'aipp_c')
    params.lambda = 0.9 / params.m;
    params.i_increase_lambda = false;
  elseif strcmp(aipp_type, 'aipp_v1') % decreasing ONLY
    params.lambda = 1;
    params.i_increase_lambda  = false;
  elseif strcmp(aipp_type, 'aipp_v2')
    params.lambda =  1 / params.m;
    params.i_increase_lambda = true;
  end
  
  % mu_fn -----------------------------------------> dependent on aipp_type
  if strcmp(aipp_type, 'aipp')
    params.mu_fn = @(lambda) max(0, 1 - lambda * params.m);
  elseif strcmp(aipp_type, 'aipp_c')
    params.mu_fn = @(lambda) max(0, 1 - lambda * params.m);
  elseif strcmp(aipp_type, 'aipp_v1') % decreasing ONLY
    params.mu_fn = @(lambda) 1;
  elseif strcmp(aipp_type, 'aipp_v2')
    params.mu_fn = @(lambda) 1;
  end
  
  % tau_mult = 10
  if(~isfield(params, 'tau_mult') && strcmp(aipp_type, 'aipp'))
    params.tau_mult = params.sigma;
  elseif(~isfield(params, 'tau_mult'))
    params.tau_mult = 10;
  end
 
  % tau = tau_mult * (lambda * M + 1) in the general case
  if(~isfield(params, 'tau') && ~strcmp(aipp_type, 'aipp'))
    params.tau = params.tau_mult * (params.lambda * params.M + 1);
    params.i_variable_tau = true;
  else
    params.i_variable_tau = false;
  end
  
  % acg_steptype = "variable" (can be set to be "constant")
  if (~isfield(params, 'acg_steptype'))
    params.acg_steptype = "variable";
  end
  
  % i_logging = false
  if (~isfield(params, 'i_logging')) 
    params.i_logging = false;
  end

end