%{

FILE DATA
---------
Last Modified: 
  May 31, 2021
Coders: 
  Weiwei Kong

%}

function [model, history] = HiAPeM(~, oracle, params)
% A hybrid inexact augmented Lagrangian and penalty method (HiAPeM) for 
% solving a nonconvex composite optimization model with nonlinear 
% convex constraints, i.e., g(x) <= 0 and Ax=b.
% 
% Note:
% 
%   Based on the paper:
%
%   Li, Z., & Xu, Y. (2020). Augmented lagrangian based first-order methods 
%   for convex and nonconvex programs: nonergodic convergence and iteration
%   complexity. *arXiv preprint arXiv:2003.08880*\.
%
% Arguments:
% 
%   ~ : The first argument is ignored
% 
%   oracle (Oracle): The oracle underlying the optimization problem.
% 
%   params (struct): Contains instructions on how to call the framework.
% 
% Returns:
%   
%   A pair of structs containing model and history related outputs of the 
%   solved problem associated with the oracle and input parameters.
%

  % Timer start.
  t_start = tic;
  
  % Initialize constant params.
  time_limit = params.time_limit;
  iter_limit = params.iter_limit;
  
  % Set the topology.
  norm_fn = params.norm_fn;
  prod_fn = params.prod_fn;
  
  % Fill in OPTIONAL input params (custom implementation).
  params = set_default_params(params);
   
  % Initialize basic algorithm params.
  rho = params.rho;
  sigma = params.sigma;
  beta0 = params.beta0;
  x0 = params.x0;
  K0 = params.N0;
  N1 = params.N1;
  gamma = params.gamma;
  L_min = params.L_min;
  eps = params.eps;
    
  % Iterate.
  iter = 0; % Set to 0 because it calls an inner subroutine.
  stage = 1;
  while true
    
    % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit)
      break;
    end
    
    % =====================================================================
    % ------------------------------ PHASE I ------------------------------
    % =====================================================================
    for k=0:(K0-1)
                 
      % Call the iALM to compute (beta, x, p)
      iALM_params = params;
      iALM_params.eps = params.eps1Hat / 2;
      iALM_params.t_start = t_start;
      iALM_params.x0 = x0;
      [iALM_model, iALM_history] = iALM(oracle, iALM_params);
      iter = iter + iALM_history.iter;
      beta = iALM_model.beta;
      x = iALM_model.x;
      y = iALM_model.y;
      z = iALM_model.z;
      
      % Early termination.
      iALM_resid = norm_fn(x - x0);
      % DEBUG
%       disp(table(iALM_resid, eps / (4 * rho), iter));
      if (norm_fn(x - x0) <= eps / (4 * rho))
        break
      end
      
      % Update iterates
      y0 = y;
      z0 = z;
      x0 = x;
    end 
    
    % DEBUG
    break
    
    % ====================================================================
    % ----------------------------- PHASE II -----------------------------
    % ====================================================================
    
  end
  
  % Get ready to output
  model.x = x;
  model.v = 0.00;
  model.w = 0.00;
  history.iter = iter;
  history.runtime = toc(t_start);
  
end

%% Subroutines

function [model, history] = iALM(oracle, params)
% iALM subroutine used in the HiAPeM.

  % Parse algorithm inputs.
  rho = params.rho;
  eps = params.eps;
  sigma = params.sigma;
  xBar = params.x0;
  y0 = zeros(size(params.lin_constr_fn(xBar)));
  z0 = zeros(size(params.nonlin_constr_fn(xBar)));
  beta0 = params.beta0;
  
  % Solver params.
  norm_fn = params.norm_fn;
  t_start = params.t_start;
  time_limit = params.time_limit;
  iter_limit = params.iter_limit;
  
  % For ease of notation, we let c1(x) := Ax-b and c2(x) := f(x), with
  % constraints c1(x)=0 and c2(x) in -K.
  
  % Initialize special constraint functions
  norm_fn = params.norm_fn;
  prod_fn = params.prod_fn;
  pc_fn = params.set_projector; % projection onto the cone K.
  
  % Cone projector function on the point q2 = p2 + beta * c2(x) on the 
  % cones: -K (primal) and K^* (dual).
  function [dual_point, primal_point]= cone_proj(x, p2, beta)
    p_step = p2 + beta * params.nonlin_constr_fn(x);
    primal_point = -pc_fn(-p_step);
    dual_point = p_step - primal_point;
  end

  % Special gradient operator. Evaulates grad_f(x)(y).
  function out_grad = grad_eval(grad_eval_fn, x, y)
    if nargin(grad_eval_fn) == 1
      out_grad = tsr_mult(grad_eval_fn(x), y, 'dual');
    elseif nargin(grad_eval_fn) == 2
      out_grad = grad_eval_fn(x, y);
    else
      error(...
        ['Unknown function prototype for the gradient of the ', ...
         'constraint function']);
    end
  end
  
  % Function that creates the proxified augmented Lagrangian oracle 
  % for the function
  %
  %   L_c(x; p1, p2) := 
  %     phi(x) + ...
  %     <p1, c1(x)> + beta / 2 * |c1(x)| ^ 2 +
  %     1 / (2 * beta) * [dist(p2 + beta * c2(x), -K) - |p2| ^ 2],
  %
  % where K is the primal cone.
  function al_oracle = create_al_oracle(p1, p2, beta)
    
    % Create the penalty oracle.
    function oracle_struct = alp_eval_fn(x)
      dual_proj_proint1 = p1 + beta * params.lin_constr_fn(x);
      [dual_proj_proint2, primal_proj_point2]= cone_proj(x, p2, beta);
      p_step = p2 + beta * params.nonlin_constr_fn(x);
      dist_val = norm_fn((-p_step) - primal_proj_point2);
      % Auxiliary function values.
      oracle_struct.f_s = @() ...
        prod_fn(p1, params.lin_constr_fn(x)) + ...
        beta / 2 * norm_fn(params.lin_constr_fn(x)) ^ 2 + ...
        1 / (2 * beta) * (dist_val ^ 2 - norm_fn(p2) ^ 2);
      oracle_struct.f_n = @() 0;
      % Auxiliary gradient operator.
      oracle_struct.grad_f_s = @() ...
          grad_eval(params.lin_grad_constr_fn, x, dual_proj_proint1) + ...
          grad_eval(params.nonlin_grad_constr_fn, x, dual_proj_proint2);
      oracle_struct.prox_f_n = @(lam) x;
    end
    oracle_AL1 = Oracle(@alp_eval_fn);
    
    % Create the combined oracle.
    al_oracle = copy(oracle);
    al_oracle.add_smooth_oracle(oracle_AL1);
   
  end
  
  % Initialize the invariant APG params.
  apg_params = params;
  apg_params.mu = rho;
  apg_params.eps = ...
    eps / 2 * sqrt((sigma - 1) / (sigma + 1)) * min([1, sqrt(rho)]);
    
%   % DEBUG
%   yOrig = y0;
%   zOrig = z0;
%   betaOrig = beta0;
  
  % Iterate.
  iter = 0; % Set to 0 because it calls an inner subroutine.
  x0 = xBar;
  x = xBar;
  y = y0;
  z = z0;
  quad_f_n = @(u) 0;
  quad_prox_f_n = @(u, lam) x;
  quad_f_s = @(u) rho * norm_fn(u - xBar) ^ 2;
  grad_quad_f_s = @(u) 2 * rho * (u - xBar);
  quad_oracle = Oracle(quad_f_s, quad_f_n, grad_quad_f_s, quad_prox_f_n);
  while true
        
    % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit)
      break;
    end
    
    % Create the APG oracle.
    apg_oracle = create_al_oracle(y0, z0, beta0);
    apg_oracle.add_smooth_oracle(quad_oracle);
        
    % Call the APG and parse the outputs.
    apg_params.x0 = x0;
    [apg_model, apg_history] = AdapAPG(apg_oracle, apg_params);
    x = apg_model.x;
    iter = iter + apg_history.iter;
    
    % Update multipliers
    y = y0 + beta0 * params.lin_constr_fn(x);
    z_step = z0 + beta0 * params.nonlin_constr_fn(x);
    z = z_step - params.set_projector(-z_step);
    
    % Check for termination.
    err1 = (norm_fn([y0; z0]) + norm_fn([y; z])) / beta0; 
    err2 = abs(z' * params.nonlin_constr_fn(x));
    err = max([err1, err2]);
%     % DEBUG
%     apg_iter = apg_history.iter;
%     disp(table(iter, apg_iter, err, eps, beta0));
    if (err <= eps)
      break;
    end
    
    % Update iterates
    x0 = x;
    y0 = y;
    z0 = z;
    beta0 = sigma * beta0;
    
  end
  
  % Get ready to output
  model.x = x;
  model.y = y;
  model.z = z;
  model.beta = beta0;
  history.iter = iter;
  
end

function [model, history] = AdapAPG(oracle, params)
% Adaptive Accelerated Proximal Gradient (APG) Method used in the iALM.

  % Parse algorithm inputs.
  eps = params.eps;
  L_min = params.L_min;
  gamma1 = params.gamma1;
  gamma2 = params.gamma2;
  x_m2 = params.x0;
  prod_fn = params.prod_fn;
  
  % Solver params.
  norm_fn = params.norm_fn;
  t_start = params.t_start;
  time_limit = params.time_limit;
  iter_limit = params.iter_limit;
  
  % Special oracles.
  o_at_x_m2 = copy(oracle);
  o_at_x_m1_next = copy(oracle);
  o_at_x_m1 = copy(oracle);
  
  % First loop to estimate L0.
  L0 = L_min;
  o_at_x_m2.eval(x_m2);
  iter = 1;
  while true
       
    % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit)
      break;
    end
    
    % Compute L0 and x_m1.
    L0 = gamma1 * L0;
    o_at_x_m1_next.eval(x_m2 - (1 / L0) * o_at_x_m2.grad_f_s());
    x_m1 = o_at_x_m1_next.prox_f_n(1 / L0);
    
    % Check termination
    o_at_x_m1.eval(x_m1);
    if (o_at_x_m1.f_s() <= o_at_x_m2.f_s() + ...
        prod_fn(o_at_x_m2.grad_f_s(), x_m1 - x_m2) + ...
        L0 / 2 * norm_fn(x_m1 - x_m2) ^ 2)
      break;
    end
    
    % Update iterates
    iter = iter + 1;
    
  end
  
  % Main APG loop.
  L = L0;
  x = x_m1;
  x0 = x_m1;
  alpha0 = 1;
  o_at_tx = copy(oracle);
  o_at_x_p1 = copy(oracle);
  ALS_oracle = copy(oracle);
  ALS_params = params;
  while true
        
    % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit)
      break;
    end
    
    % Line search subroutine.
    ALS_params.x = x;
    ALS_params.x0 = x0;
    ALS_params.L = L;
    ALS_params.alpha0 = alpha0;
    [ALS_model, ALS_history] = AccelLineSearch(ALS_oracle, ALS_params);
    
    % Parse output.
    x_p1 = ALS_model.x_p1;
    M = ALS_model.M;
    alpha = ALS_model.alpha;
    tx = ALS_model.tx;
    L = max([L_min, M / gamma2]);
        
    % Check for termination. Here, v is a proxy for dist(0, ∂F(x)).
    o_at_tx.eval(tx);
    o_at_x_p1.eval(x_p1);
    v = L * (tx - x_p1) + o_at_x_p1.grad_f_s() - o_at_tx.grad_f_s();
    if (norm_fn(v) <= eps)
      break;
    end

    % Update iterates
    x0 = x;
    x = x_p1;
    alpha0 = alpha;
    iter = iter + ALS_history.iter;
    
  end
  
  % Get ready to output.
  model.x = x_p1;
  history.iter = iter;
  
end

function [model, history] = AccelLineSearch(oracle, params)
% Line search subroutine used in AdapAPG

  % Global constants
  MACHEPS = 1e-3; % buffer room for comparing quantities

  % Parse inputs.
  x = params.x;
  x0 = params.x0;
  L = params.L;
  mu = params.mu;
  alpha0 = params.alpha0;
  gamma1 = params.gamma1;
  prod_fn = params.prod_fn;
  norm_fn = params.norm_fn;
  
  % Initialization.
  o_at_tx = copy(oracle);
  o_at_tx_next = copy(oracle);
  o_at_x_p1 = copy(oracle);
  
  % Main loop.
  M = L / gamma1; % Equivalent to L in the paper.
  iter = 1;
  while true
    M = M * gamma1;
    alpha = sqrt(mu / M);
    % Compute and evaluate tx_k (= y_k in the paper).
    tx = x + (alpha * (1 - alpha0)) / (alpha0 * (1 + alpha)) * (x - x0);
    o_at_tx.eval(tx);
    % Compute and evaluate x_{k+1}.
    o_at_tx_next.eval(tx - (1 / M) * o_at_tx.grad_f_s());
    x_p1 = o_at_tx_next.prox_f_n(1 / M);
    o_at_x_p1.eval(x_p1);
    
%     % DEBUG
%     disp(table(...
%       iter, ...
%       norm_fn(x0 - x), ...
%       norm_fn(x - x_p1), ...
%       norm_fn(x0 - x_p1)));    
    
    % Check the descent condition in a relative sense.
    small_err = o_at_x_p1.f_s();
    large_err = ...
      o_at_tx.f_s() + prod_fn(o_at_tx.grad_f_s(), x_p1 - tx) + ...
      M / 2 * norm_fn(x_p1 - tx) ^ 2;
    base_err = max(abs([small_err, large_err, MACHEPS]));
    if ((small_err - large_err) / base_err < MACHEPS)
      break;
    end
    
%     % DEBUG
%     true_M = params.M;
%     beta0 = params.beta0;
%     K_constr = params.K_constr;
%     rho = params.rho;
%     Lpsc = true_M + beta0 * K_constr ^ 2 + rho;
%     if (M > Lpsc)
%       disp(table(M, true_M, beta0, rho, Lpsc, iter));
%       disp(table(o_at_x_p1.f_s(), ...
%                   o_at_tx.f_s() + ...
%                   prod_fn(o_at_tx.grad_f_s(), x_p1 - tx) + ...
%                   M / 2 * norm_fn(x_p1 - tx) ^ 2));
%     end    
    
    % Update iterates
    iter = iter + 1;
  end  

  % Prepare output.
  model.x_p1 = x_p1;
  model.M = M;
  model.alpha = alpha;
  model.tx = tx;
  history.iter = iter;
end

%% Helper functions

% Fills in parameters that were not set as input.
function params = set_default_params(params)

  % Overwrite if necessary.
  if (~isfield(params, 'rho')) 
    params.rho = params.m;
  end
  if (~isfield(params, 'L_min')) 
    params.L_min = 2 * params.m;
  end
  if (~isfield(params, 'gamma1')) 
    params.gamma1 = 2;
  end
  if (~isfield(params, 'gamma2')) 
    params.gamma2 = 1.25;
  end
  if (~isfield(params, 'N0')) 
    params.N0 = 10;
  end
  if (~isfield(params, 'N1')) 
    params.N1 = 2;
  end
  if (~isfield(params, 'sigma')) 
    params.sigma = 3;
  end
  if (~isfield(params, 'beta0')) 
    params.beta0 = max([1, params.L / params.K_constr ^ 2]);
  end
  if (~isfield(params, 'gamma')) 
    params.gamma = 1.1;
  end
  if (~isfield(params, 'eps'))
    params.eps = min([params.opt_tol, params.feas_tol]);
  end
  if (~isfield(params, 'eps1Hat')) 
    params.eps1Hat = params.eps;
  end
  if (~isfield(params, 'eps2Hat')) 
    params.eps2Hat = ...
      params.eps / (2 * sqrt(2)) * min([1, 1 / sqrt(params.rho)]);
  end
  % Partition default constraint functions
  if (~isfield(params, 'lin_constr_fn')) % c1(x) := Ax - b
    params.lin_constr_fn = params.constr_fn; 
  end
  if (~isfield(params, 'lin_grad_constr_fn')) 
    params.lin_grad_constr_fn = params.grad_constr_fn; 
  end
  if (~isfield(params, 'nonlin_constr_fn')) % c2(x) := f(x)
    params.nonlin_constr_fn = params.constr_fn; 
  end
  if (~isfield(params, 'nonlin_grad_constr_fn')) 
    params.nonlin_grad_constr_fn = params.grad_constr_fn; 
  end
  
end