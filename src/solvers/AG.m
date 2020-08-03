%{

A variant of the accelerated gradient (AG) Method that is adapted from 
the paper (Algorithm 2):

"Accelerated gradient methods for nonconvex nonlinear and stochastic 
programming", DOI: 10.1007/s10107-015-0871-8.

Specifically, this variant replaces the Lipschitz constant Lf with the
upper curvature M.

VERSION 1.0
-----------

Input
------
oracle:
  An Oracle object.
params:
  Input parameters of this function.

Output
------
model:
  A struct array containing model related outputs.
history:
  A struct array containing history (e.g. iteration counts, runtime, etc.)
  related outputs.

%}

function [model, history] = AG(oracle, params)

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
  
end