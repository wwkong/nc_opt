%{

FILE DATA
---------
Last Modified: 
  August 2, 2020
Coders:
  Weiwei Kong

%}

function [model, history] = AG(oracle, params)
% The accelerated gradient (AG) method. 
%
% See Also: 
%
%   **src.solvers.UPFAG**
% 
% Note:
% 
%   A variant of Algorithm 2 (with the upper curvature $M$ replacing the 
%   Lipschitz constant $L_f$) from the paper:
%
%   Ghadimi, S., & Lan, G. (2016). Accelerated gradient methods for nonconvex 
%   nonlinear and stochastic programming. *Mathematical Programming, 
%   156*\(1-2), 59-99.
%
% Arguments:
%
%   oracle (Oracle): The oracle underlying the optimization problem.
%
%   params (struct): Contains instructions on how to call the algorithm.
%
% Returns:
%   
%   A pair of structs containing model and history related outputs of the 
%   solved problem associated with the oracle and input parameters.
%

  % Timer start
  t_start = tic;
    
  % Initialize
  x0 = params.x0;
  x_prev = x0;
  xAG_prev = x0;
  M = params.M;
  norm_fn = params.norm_fn;
  beta = 1 / (2 * M);
  iter = 0;
  k = 0;
  
  % Fill in OPTIONAL input params.
  params = set_default_params(params);
  
  % History params.
  if params.i_logging
    oracle.eval(x0);
    function_values = oracle.f_s() + oracle.f_n();
    iteration_values = 0;
    time_values = 0;
  end
  
  % Solver params
  opt_tol = params.opt_tol;
  time_limit = params.time_limit;
  iter_limit = params.iter_limit;
     
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
        
    % Set step sizes
    k = k + 1;
    alphaK = 2 / (k + 1);
    lambdaK = k * beta / 2;
    
    % Oracle at xMD
    xMD = (1 - alphaK) * xAG_prev + alphaK * x_prev;
    o_xMD = oracle.eval(xMD);
    grad_f_s_at_xMD = o_xMD.grad_f_s();
    
    % Oracle at xNext
    o_xNext = oracle.eval(x_prev - grad_f_s_at_xMD * lambdaK);
    x = o_xNext.prox_f_n(lambdaK);
    
    % Oracle at xMD_next
    o_xMD_next = oracle.eval(xMD - grad_f_s_at_xMD * beta); 
    xAG = o_xMD_next.prox_f_n(beta);
    
    % Oracle at xAG
    if (isfield(o_xMD_next, 'grad_f_s_at_prox_f_n'))
      grad_f_s_at_xAG = o_xMD_next.grad_f_s_at_prox_f_n(beta);
    else
      o_xAG = oracle.eval(xAG);
      grad_f_s_at_xAG = o_xAG.grad_f_s();
    end
    
    % Check for termination
    G = - 1 / beta * (xAG - xMD);
    v_bar = grad_f_s_at_xAG - grad_f_s_at_xMD + G;
    if (norm_fn(v_bar) <= opt_tol)
      break
    end
    
    % Update history
    if params.i_logging
      oracle.eval(xMD);
      function_values(end + 1) = oracle.f_s() + oracle.f_n();
      iteration_values(end + 1) = iter;
      time_values(end + 1) = toc(t_start);
    end
    
    % Update iterates
    x_prev = x;
    xAG_prev = xAG;
    iter = iter + 1;
    
  end
  
  % Get ready to output
  runtime = toc(t_start);
  
  % Parse and output
  model.x = xMD;
  model.v = v_bar;
  history.runtime = runtime;
  history.iter = iter;
  if params.i_logging
    history.function_values = function_values;
    history.iteration_values = iteration_values;
    history.time_values = time_values;
  end
  
end

% Fills in parameters that were not set as input.
function params = set_default_params(params)

  % Overwrite if necessary.
  if (~isfield(params, 'i_logging')) 
    params.i_logging = false;
  end

end