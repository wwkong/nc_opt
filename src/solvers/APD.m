% SPDX-License-Identifier: MIT
% Copyright © 2022 Weiwei "William" Kong

function [model, history] = APD(oracle, params)
% Variants of the accelerated proximal descent (APD) method
%
% See Also:
%
%   **src/solvers/ACG.m**
%
% Arguments:
%
%   oracle (Oracle): The oracle underlying the optimization problem.
%
%   params.theta (double): Determines the accuracy of the inner ACG call. Defaults to ``4``.
%
%   params.alpha (double): Determines how quickly the line search subroutine of m0 (multiplicatively)
%     increases its estimate. Defaults to ``3``.
%
%   params.beta (double): Determines how quickly the line search subroutine of the ACG call (multiplicatively)
%     increases its estimate. Defaults to ``3``.
%
%   params.M0 (function handle): Initial upper curvature estimate. Defaults to ``1``.
%
%   params.m0 (function handle): Initial upper lower estimate. Defaults to ``1``.
%
% Returns:
%   
%   A pair of structs containing model and history related outputs of the solved problem associated with the oracle and input
%   parameters.
%
  
  % Start the timer.
  t_start = tic;
  
  % PRE-PROCESSING
  
  % Set REQUIRED input params.
  z0 = params.x0;
  norm_fn = params.norm_fn;
  opt_tol = params.opt_tol; 
  
  % Fill in OPTIONAL input params.
  params = set_default_params(params);
  
  % Set other input params.
  m = params.m0;
  M = params.M0;
  
  % Solver params
  iter_limit = params.iter_limit;
  time_limit = params.time_limit;
  
  % Initialize the model struct in case the ACG fails
  x = z0;
  v = -Inf;
  
  % Initialize some auxillary functions and constants.
  iter = 0;
  outer_iter = 1;
  params_acg = params;
  params_acg.mu = 1.0 / 2.0;
  params_acg.sigma = 1 / 4.0; 
  params_acg.init_mult_L = 1.0;
  params_acg.mult_L = params.beta;
  params_acg.termination_type = 'apd';
  params_acg.acg_steptype = 'variable';
  params_acg.eta_type = 'accumulative';
  params_acg.L = Inf; % Upper bound of L_est.
     
  %% MAIN ALGORITHM

  while true
       
    % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit)
      break;
    end
    
    %% ACG CALL AND POST-PROCESSING
    
    % Set up the ACG oracle.
    oracle_acg = copy(oracle);
    oracle_acg.proxify(1 / (2 * m), z0);
    
    % Set ACG parameters.
    params_acg.x0 = z0;
    params_acg.z0 = z0;
    params_acg.L_est = M / (2 * m) + 1;
    params_acg.time_limit = max([0, time_limit - toc(t_start)]);
        
    % Repeatedly call the ACG.
    if strcmp(params.line_search_type, 'optimistic')
      L_acg = M / (2 * m) + 1;
      params_acg.init_mult_L = max(1, L_acg / ((1 + params.beta) / 2)) / L_acg;
    end
    [model_acg, history_acg] = ACG(oracle_acg, params_acg);
    iter = iter + history_acg.iter;
    if (model_acg.status < 0)
      m = params.alpha * m;
      continue;
    end     
      
    % Check for termination.
    x = model_acg.y;
    v = 2 * m * (model_acg.u + z0 - x);
    if(norm_fn(v) <= opt_tol)
      break;
    end
    
    % Update iterates
    M = (model_acg.L_est - 1) * (2 * m);
    if strcmp(params.line_search_type, 'optimistic')
      m = max(params.m0, m / ((1 + params.alpha) / 2));
    end
    z0 = x;
    outer_iter = outer_iter + 1;
  end 
  
  % Prepare the model and history
  model.x = x;
  model.v = v;
  history.iter = iter;
  history.outer_iter = outer_iter;
  history.runtime = toc(t_start);

end % function end
 
%% Helper Functions

% Fills in parameters that were not set as input
function params = set_default_params(params)
  
  % M0 = 1
  if(~isfield(params, 'm0'))
    params.M0 = 1;
  end

  % m0 = 1
  if(~isfield(params, 'm0'))
    params.m0 = 1E-4;
  end
  
  % theta = 4
  if(~isfield(params, 'theta'))
    params.theta = 1;
  end

  % alpha = 3
  if (~isfield(params, 'alpha'))
    params.alpha = 3;
  end
  
  % beta = 3
  if (~isfield(params, 'beta'))
    params.beta = 3;
  end
  
  % line_search_type = 'monotonic'
  if (~isfield(params, 'line_search_type'))
    params.line_search_type = 'monotonic';
  end
  
  % i_logging = false
  if (~isfield(params, 'i_logging')) 
    params.i_logging = false;
  end
  
  % i_debug = false
  if (~isfield(params, 'i_debug')) 
    params.i_debug = false;
  end

end
