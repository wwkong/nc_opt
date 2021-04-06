%{

FILE DATA
---------
Last Modified: 
  August 19, 2020
Coders: 
  Weiwei Kong, Jiaming Liang

%}  

function [model, history] = NC_FISTA(oracle, params)
% The nonconvex fast iterative soft theresholding (ADAP-NC-FISTA) method. 
% 
% See Also:
%
% **src.solvers.ADAP_FISTA**
% 
% Note:
% 
%   Based on the paper (see the NC-FISTA method):
%
%   Liang, J., Monteiro, R. D., & Sim, C. K. (2019). A FISTA-type accelerated 
%   gradient algorithm for solving smooth nonconvex composite optimization 
%   problems. *arXiv preprint arXiv:1905.07010*.
%
% Arguments:
%
%   oracle (Oracle): The oracle underlying the optimization problem. 
%
%   params.lambda (double): The first stepsize used by the method (see 
%     $\lambda$ from the original paper). Defaults to ``0.99 / L``.
%
%   params.xi (double): Determines $A_0$ from the original paper (see $\xi$
%     from the original paper). Defaults to ``1.05 * m``.
%
% Returns: 
%   
%   A pair of structs containing model and history related outputs of the 
%   solved problem associated with the oracle and input parameters.
%    

  % Timer start
  t_start = tic;
  
  % Main params
  m = params.m;
  z0 = params.x0;
  norm_fn = params.norm_fn;
  
  % Fill in OPTIONAL input params.
  params = set_default_params(params);
  lambda = params.lambda;
  xi = params.xi;
  
  % Solver params.
  opt_tol = params.opt_tol;
  time_limit = params.time_limit;
  iter_limit = params.iter_limit;
  
  % History params.
  if params.i_logging
    oracle.eval(z0);
    function_values = oracle.f_s() + oracle.f_n();
    iteration_values = 0;
    time_values = 0;
    vnorm_values = Inf;
  end
  history.min_norm_of_v = Inf;
  
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
    tau = 2*xi*lambda/a;
    tx = A/ANext*y + a/ANext*x;
    o_tx = oracle.eval(tx);
    grad_f_tx = o_tx.grad_f_s();
    o_tx_next = oracle.eval(tx - lambda / (1 + tau) * grad_f_tx);
    yNext = o_tx_next.prox_f_n(lambda/(1+tau));
    xNext = (1+tau)*ANext/(a*(tau*a+1))*yNext - A/(a*(tau*a+1))*y;   
    o_yNext = oracle.eval(yNext);
    grad_f_yNext = o_yNext.grad_f_s();
    v = (1 + tau) / lambda * (tx - yNext) + grad_f_yNext - grad_f_tx;
    
    % Check for termination
    norm_v = norm_fn(v);
    history.min_norm_of_v = min([history.min_norm_of_v, norm_v]);
    if (norm_fn(v) <= opt_tol)
      break
    end
    
    % Update history
    if params.i_logging
      oracle.eval(xNext);
      function_values(end + 1) = oracle.f_s() + oracle.f_n();
      iteration_values(end + 1) = iter;
      time_values(end + 1) = toc(t_start);
      vnorm_values(end + 1) = norm_v;
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
  if params.i_logging
    history.function_values = function_values;
    history.iteration_values = iteration_values;
    history.time_values = time_values;
    history.vnorm_values = vnorm_values;
  end
  
end % function end

% Fills in parameters that were not set as input.
function params = set_default_params(params)

  % Overwrite if necessary.
  if (~isfield(params, 'lambda')) 
    params.lambda = 0.99 / params.L;
  end
  if (~isfield(params, 'xi')) 
    params.xi = 1.05 * params.m;
  end
  if (~isfield(params, 'i_logging')) 
    params.i_logging = false;
  end

end