%{

The well-known exact composite gradient (ECG) method with constant 
stepsize. For reference, see the paper:

"Gradient methods for minimizing composite functions", Mathematical 
Programming.

FILE DATA
---------
Last Modified: 
  August 2, 2020
Coders: 
  Weiwei Kong

INPUT
-----
oracle:
  An Oracle object.
params:
  A struct containing input parameters for this function.

OUTPUT
------
model:
  A struct containing model related outputs (e.g. solutions).
history:
  A struct containing history related outputs (e.g. runtimes).

%}

function [model, history] = ECG(oracle, params)

  % Timer start.
  t_start = tic;
  
  % Initialize params.
  x0 = params.x0;
  x_prev = x0;
  M = params.M;
  norm_fn = params.norm_fn;
  iter = 1;
  
  % Solver params.
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
    
    % Oracle at xPrev.
    o_xPrev = oracle.eval(x_prev);
    grad_f_s_at_x_prev = o_xPrev.grad_f_s();
    y_bar = x_prev - 1 / M * grad_f_s_at_x_prev;
    
    % Oracle at yBar.
    o_yBar = oracle.eval(y_bar);
    x_bar = o_yBar.prox_f_n(1 / M);
    
    % Oracle at xBar.
    o_xBar = oracle.eval(x_bar);
    grad_f_s_at_x_bar = o_xBar.grad_f_s();
    
    % Check for termination.
    v_bar = M * (x_prev - x_bar) + grad_f_s_at_x_bar - grad_f_s_at_x_prev;
    if (norm_fn(v_bar) <= opt_tol)
      break;
    end
    
    % Update iterates.
    x_prev = x_bar;
    iter = iter + 1;
    
  end
  
  % Parse outputs.
  model.x = x_bar;
  model.v = v_bar;
  history.iter = iter;
  history.runtime = toc(t_start);
  
end