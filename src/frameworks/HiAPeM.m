%{

FILE DATA
---------
Last Modified: 
  June 19, 2021
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
  x0 = params.x0;
  K0 = params.N0;
  N1 = params.N1;
  gamma = params.gamma;
  eps = params.eps;
    
  % Iterate.
  iter = 0; % Set to 0 because it calls an inner subroutine.

    
  % =====================================================================
  % ------------------------------ PHASE I ------------------------------
  % =====================================================================
  
  % Call iALM for K0 iterations.
  for k=0:(K0 - 1)
    
    % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit && iter > 1)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit)
      break;
    end
    
    % Call the iALM to compute (beta, x, p)
    iALM_params = params;
    iALM_params.alg_type = "iALM";
    iALM_params.eps = params.eps1Hat / 2;
    iALM_params.t_start = t_start;
    iALM_params.xBar = x0;
    [iALM_model, iALM_history] = MM_alg(oracle, iALM_params);
    iter = iter + iALM_history.iter;
    beta = iALM_model.beta;
    x = iALM_model.x;
    y = iALM_model.y;
    z = iALM_model.z;
    if (norm_fn(v) <= eps)
      break
    end
    % Update iterates
    x0 = x;
  end
  
  % ====================================================================
  % ----------------------------- PHASE II -----------------------------
  % ====================================================================
  
  % Call penalty method and switch to iALM at exactly iteration (Ks - 1).
  s = 1;
  t = 0;
  k = k + 1;
  Ks = K0 + N1;
  while true
    
    % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit && iter > 0)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit)
      break;
    end
    
    % Call the penalty method.
    Pen_params = params;
    Pen_params.alg_type = "PenMM";
    Pen_params.eps = params.eps2Hat / 2;
    Pen_params.t_start = t_start;
    Pen_params.beta = beta;
    Pen_params.xBar = x;
    Pen_params.yBar = y;
    Pen_params.zBar = z;
    [Pen_model, Pen_history] = MM_alg(oracle, Pen_params);
    iter = iter + Pen_history.iter;
    beta = Pen_model.beta;
    x = Pen_model.x;
    % Early termination.
    if (norm_fn(x - x0) <= eps / (4 * rho))
      break
    end
    
    % Increment counters and check iALM condition.
    t = t + 1;
    k = k + 1;
    
    if (k == (Ks - 1))
      % Call the iALM only when k is (Ks - 1).
      iALM_params = params;
      iALM_params.alg_type = "iALM";
      iALM_params.eps = params.eps1Hat / 2;
      iALM_params.t_start = t_start;
      iALM_params.xBar = x;
      [iALM_model, iALM_history] = MM_alg(oracle, iALM_params);
      iter = iter + iALM_history.iter;
      beta = iALM_model.beta;
      x = iALM_model.x;
      y = iALM_model.y;
      z = iALM_model.z;
      % Early termination.
      if (norm_fn(x - x0) <= eps / (4 * rho))
        break
      end
      % Multipliers and indices (s, t, k) are updated when iALM is called.
      t = 0;
      k = Ks;
      Ks = Ks + ceil(N1 * gamma ^ s);
    end
    
    % Update iterates.
    x0 = x;
  end
  
  % Get ready to output
  model.x = x;
  model.v = (x - x0) * 4 * rho;
  model.w = params.lin_constr_fn(x); % ADD NONLINEAR RESIDUAL
  history.iter = iter;
  history.runtime = toc(t_start);
  
end

%% Subroutines

function [model, history] = MM_alg(oracle, params)
% Combined subroutine that implements either iALM or PenMM in the HiAPeM.
% The choice of subroutine depends on the field 'alg_type';

  % Parse algorithm inputs.
  rho = params.rho;
  eps = params.eps;
  sigma = params.sigma;
  x0 = params.xBar;
  alg_type = params.alg_type;
  if strcmp(alg_type, 'iALM')
    y0 = zeros(size(params.lin_constr_fn(x0)));
    z0 = zeros(size(params.nonlin_constr_fn(x0)));
  elseif strcmp(alg_type, 'PenMM')
    y0 = params.yBar;
    z0 = params.zBar;
  else
    error('Unknown alg_type!');
  end
  beta = params.beta0;
  
  % Solver params.
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
  
  % Function that creates the augmented Lagrangian oracle of the function
  %
  %   L_c(x; p1, p2) := 
  %     phi(x) + ...
  %     <p1, c1(x)> + beta / 2 * |c1(x)| ^ 2 +
  %     1 / (2 * beta) * [dist(p2 + beta * c2(x), -K) - |p2| ^ 2],
  %
  % where K is the primal cone, c1 is the linear constraint function, c2 is the
  % nonlinear constraint function.
  function al_oracle = create_al_oracle(p1, p2, beta)
    
    % Create the penalty oracle.
    function oracle_struct = alp_eval_fn(x)
      dual_proj_point1 = p1 + beta * params.lin_constr_fn(x);
      [dual_proj_point2, primal_proj_point2]= cone_proj(x, p2, beta);
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
        grad_eval(params.lin_grad_constr_fn, x, dual_proj_point1) + ...
        grad_eval(params.nonlin_grad_constr_fn, x, dual_proj_point2);
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
  if strcmp(alg_type, "iALM")
    apg_params.eps = ...
      eps / 2 * sqrt((sigma - 1) / (sigma + 1)) * min([1, sqrt(rho)]);
  elseif strcmp(alg_type, "PenMM")
    apg_params.eps = eps * min([1, sqrt(rho)]);
  else
    error('Unknown alg_type!')
  end
  
  % Iterate.
  iter = 0; % Set to 0 because it calls an inner subroutine.
  outer_iter = 1;
  x = x0;
  y = y0;
  z = z0;
  beta0 = beta;
  quad_f_n = @(u) 0;
  quad_prox_f_n = @(u, lam) u;
  quad_f_s = @(u) rho * norm_fn(u - x0) ^ 2;
  grad_quad_f_s = @(u) 2 * rho * (u - x0);
  quad_oracle = Oracle(quad_f_s, quad_f_n, grad_quad_f_s, quad_prox_f_n);
  while true
        
    % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit && iter > 0)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit)
      break;
    end
    
    % Create the APG oracle.
    apg_oracle = create_al_oracle(y0, z0, beta);
    apg_oracle.add_smooth_oracle(quad_oracle);    
    
    % Call the APG and parse the outputs.
    apg_params.x0 = x0;
    apg_params.time_limit = max([0, time_limit - toc(t_start)])
    % DEBUG
    apg_params.L_max = params.M + 2 * params.rho + beta * params.K_constr ^ 2;
    if params.full_L_min
      apg_params.L_min = apg_params.L_max;
    end    
    [apg_model, apg_history] = AdapAPG(apg_oracle, apg_params);
    x = apg_model.x;
    iter = iter + apg_history.iter;
    
    % Update multipliers
    y = y0 + beta * params.lin_constr_fn(x);   
    z_step = z0 + beta * params.nonlin_constr_fn(x);
    z = z_step - params.set_projector(-z_step);
    
    % Check for termination.
    err1 = (norm_fn([y0; z0]) + norm_fn([y; z])) / beta; 
    err2 = abs(z' * params.nonlin_constr_fn(x));
    err = max([err1, err2]);    
    if (err <= eps)
      break;
    end
    
    % DEBUG
    nrm_y = norm_fn(y);
    L_min = apg_params.L_min;
    disp(table(alg_type, err, iter, outer_iter, L_min, beta, nrm_y));
    
    % Update iterates.
    outer_iter = outer_iter + 1;
    x0 = x;
    beta0 = beta;
    % Only the iALM updates the multipliers.
    if strcmp(alg_type, "iALM")
      y0 = y;
      z0 = z;
    end
    beta = sigma * beta;
    
  end
  
  % Get ready to output
  if strcmp(alg_type, "iALM")
    model.x = x;
    model.y = y;
    model.z = z;
    model.beta = beta;
  elseif strcmp(alg_type, "PenMM")
    model.x = x0;
    model.y = y0;
    model.z = z0;
    model.beta = beta0;
  else
    error('Unknown alg_type!')
  end  
  history.iter = iter;
  
end


%% Helper functions

% Fills in parameters that were not set as input.
function params = set_default_params(params)

  % Overwrite if necessary.
   if (~isfield(params, 'is_box')) 
    params.is_box = false;
  end
  if (~isfield(params, 'rho')) 
    params.rho = params.m;
  end
  if (~isfield(params, 'L_min')) 
    params.L_min = params.M; % Extremely unstable when L_min == m.
  end
  if (~isfield(params, 'full_L_min'))
    params.full_L_min = false;
  end
  if (~isfield(params, 'gamma1')) 
    params.gamma1 = 2;
  end
  if (~isfield(params, 'gamma2')) 
    params.gamma2 = 1.25;
  end
  if (~isfield(params, 'N0')) 
    params.N0 = 2;
  end
  if (~isfield(params, 'N1')) 
    params.N1 = 2;
  end
  if (~isfield(params, 'sigma')) 
    params.sigma = 2;
  end
  if (~isfield(params, 'beta0'))
%     params.beta0 = 0.01;
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