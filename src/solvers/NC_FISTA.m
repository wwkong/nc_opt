  function [model, history] = NC_FISTA(oracle, params)
                            
  % ===========================================
  % --------------- Parse Input ---------------
  % ===========================================
   
  % Timer start
  t_start = tic;
  
  % Main params
  m = params.m;
  L = max([params.M, params.m]);
  z0 = params.x0;
  lam = 0.99 / L;
  xi = 1.05 * m;
  norm_fn = params.norm_fn;
  
  % Solver params.
  opt_tol = params.opt_tol;
  time_limit = params.time_limit;
  iter_limit = params.iter_limit;
  
  % Initialize
  iter = 0;
  A = 2 * xi * (xi + m) / (xi - m) ^ 2;
  x = z0;
  y = z0;
  
  % Main loop
  while true
    
     % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit)
      break;
    end
        
    % Main computations
    a = (1+sqrt(1+4*A))/2;
    ANext = A + a;
    tau = 2*xi*lam/a;
    tx = A/ANext*y + a/ANext*x;
    o_tx = oracle.eval(tx);
    grad_f_tx = o_tx.grad_f_s();
    o_tx_next = oracle.eval(tx - lam / (1 + tau) * grad_f_tx);
    yNext = o_tx_next.prox_f_n(lam/(1+tau));
    xNext = (1+tau)*ANext/(a*(tau*a+1))*yNext - A/(a*(tau*a+1))*y;   
    o_yNext = oracle.eval(yNext);
    grad_f_yNext = o_yNext.grad_f_s();
    v = (1 + tau) / lam * (tx - yNext) + grad_f_yNext - grad_f_tx;
    
    % Check for termination
    if (norm_fn(v) <= opt_tol)
      break
    end
    
    % Update iterates
    A = ANext;
    x = xNext;
    y = yNext;
    iter = iter + 1;
    
  end
  
  model.A = A;
  model.x = x;
  model.y = y;
  model.v = v;
  history.runtime = toc(t_start);
  history.iter = iter;
  
end % function end