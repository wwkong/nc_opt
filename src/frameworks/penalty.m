%{

A penalty-based framework for solving a nonconvex composite optimization
model with linear set constraints, i.e., 

min  [phi(x) := f_s(x) + f_n(x)]
s.t. (A * x) in S,

where a projector onto S is readily available. The penalty parameter 'c'
for this variant starts at c = M / ||A|| ^ 2 and a warm-start routine
is applied for the starting point between cycles.


FILE DATA
---------
Last Modified: 
  August 5, 2020
Coders: 
  Weiwei Kong

INPUT
-----
solver:
  A solver for unconstrained composite optimization.
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

function [model, history] = penalty(solver, oracle, params)

  % Global constants.
  MIN_PENALTY_CONST = 1;

  % Timer start.
  t_start = tic;
  
  % Initialize constant params.
  feas_tol = params.feas_tol;
  time_limit = params.time_limit;
  iter_limit = params.iter_limit;
  o_oracle = oracle;
  
  % Initialize helper functions.
  penalty_n = @(x) 0;
  penalty_prox = @(x, lam) x;
  % Value of the feasibility function: 
  % feas(x) = |constr_fn(x) - set_projector(constr_fn(x))|. 
  function [feas, feas_vec] = feas_fn(x)
    constr_fn_vec = params.constr_fn(x);
    feas_vec = ...
      constr_fn_vec - params.set_projector(constr_fn_vec);
    feas = params.norm_fn(feas_vec);
  end
  % Value of the function (1 / 2) * feas(x) ^ 2.
  function sqr_feas = sqr_feas_fn(x)
    sqr_feas = (1 / 2) * feas_fn(x) ^ 2;
  end
  % Gradient of the function (1 / 2) * feas(x) ^ 2.
  function grad_sqr_feas = grad_sqr_feas_fn(x)
    [~, feas_vec] = feas_fn(x);
    grad_constr_fn = params.grad_constr_fn;
    %  If the gradient function has a single argument, assume that the
    %  gradient at a point is a constant tensor.
    if nargin(grad_constr_fn) == 1
      grad_sqr_feas = ...
        tsr_mult(grad_constr_fn(x), feas_vec, 'dual');
    % Else, assume that the gradient is a bifunction; the first argument is
    % the point of evaluation, and the second one is what the gradient
    % operator acts on.
    elseif nargin(grad_constr_fn) == 2
      grad_sqr_feas = grad_constr_fn(x, feas_vec);
    else
      error(...
        ['Unknown function prototype for the gradient of the ', ...
         'constraint function']);
    end
  end

  % Initialize solver parameters.
  iter = 0;
  solver_params = params;
  c = max([MIN_PENALTY_CONST, params.M / params.K_constr ^ 2]);

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
    
    % Create the penalty oracle object.
    penalty_s = @(x) c * sqr_feas_fn(x);
    grad_penalty = @(x) c * grad_sqr_feas_fn(x);
    penalty_oracle = ...
      Oracle(penalty_s, penalty_n, grad_penalty, penalty_prox);
    
    % Create the main oracle and update the model.
    % NOTE: We need to explicitly call copy() because the 'Oracle' class
    % inherits from the 'handle' class.
    solver_oracle = copy(o_oracle);
    solver_oracle.add_smooth_oracle(penalty_oracle)
    
    % Update curvatures and call the solver.
    solver_params.M = params.M + c * params.K_constr ^ 2;
    [solver_model, solver_history] = ...
      solver(solver_oracle, solver_params);
    
    % Check for termination.
    if (feas_fn(solver_model.x) <= feas_tol)
      break;
    end
    
    % Apply warm-start onto the initial point fed into the next iteration.
    solver_params.x0 = solver_model.x;

    % Update iterates.
    c = 2 * c;
    iter = iter + solver_history.iter;    
  end
  
  % Prepare to output
  [~, feas_vec] = feas_fn(solver_model.x);
  model = solver_model;
  model.y = c * feas_vec;
  model.w = feas_vec;
  history = solver_history;
  history.iter = iter;
  history.runtime = toc(t_start);
  
end