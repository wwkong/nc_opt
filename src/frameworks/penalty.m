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
constr_comp_model:
  An ConstrCompModel object.

OUTPUT
------
constr_comp_model:
  The optimized ConstrCompModel object from the input.

%}

function constr_comp_model = penalty(constr_comp_model)

  % Timer start.
  t_start = tic;
  
  % Initialize constant params.
  feas_tol = constr_comp_model.internal_feas_tol;
  time_limit = constr_comp_model.time_limit;
  iter_limit = constr_comp_model.iter_limit;
  o_constr_comp_model = constr_comp_model;
  o_oracle = constr_comp_model.oracle;
  
  % Set the flags carefully.
  constr_comp_model.i_update_oracle = false;
  constr_comp_model.i_reset = false;  
  
  % Initialize helper functions.
  penalty_n = @(x) 0;
  penalty_prox = @(x, lam) x;
  % Value of the feasibility function: 
  % feas(x) = |constr_fn(x) - set_projector(constr_fn(x))|. 
  function [feas, feas_vec] = feas_fn(x)
    constr_fn_vec = constr_comp_model.constr_fn(x);
    feas_vec = ...
      constr_fn_vec - constr_comp_model.set_projector(constr_fn_vec);
    feas = constr_comp_model.norm_fn(feas_vec);
  end
  % Value of the function (1 / 2) * feas(x) ^ 2.
  function sqr_feas = sqr_feas_fn(x)
    sqr_feas = (1 / 2) * feas_fn(x) ^ 2;
  end
  % Gradient of the function (1 / 2) * feas(x) ^ 2.
  function grad_sqr_feas = grad_sqr_feas_fn(x)
    grad_constr_fn = constr_comp_model.grad_constr_fn;
    [~, feas_vec] = feas_fn(x);
    grad_sqr_feas = grad_constr_fn(feas_vec) * feas_vec;
  end

  % Initialize solver parameters.
  iter = 0;
  c = constr_comp_model.M / constr_comp_model.K_constr ^ 2;
  constr_comp_model.update_tolerances;

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
    % inherits from the 'handle' class
    combined_oracle = copy(o_oracle);
    combined_oracle.add_smooth_oracle(penalty_oracle)
    constr_comp_model.oracle = combined_oracle;
    
    % Update curvatures and solver params.
    constr_comp_model.M = ...
      o_constr_comp_model.M + c * o_constr_comp_model.K_constr ^ 2;
    constr_comp_model.update_curvatures;
    constr_comp_model.update_solver_inputs;
    
    % Call the solver.
    constr_comp_model.call_solver;
    constr_comp_model.post_process;
    constr_comp_model.get_status;
    
    % Check for termination.
    if (feas_fn(constr_comp_model.x) <= feas_tol)
      break;
    end
    
    % Apply warm-start
    constr_comp_model.x0 = constr_comp_model.x;

    % Update iterates.
    c = 2 * c;
    o_constr_comp_model.iter = ...
      o_constr_comp_model.iter + constr_comp_model.iter;    
  end
  
  % Restore some original settings.
  constr_comp_model.update_curvatures;
  constr_comp_model.i_update_oracle = o_constr_comp_model.i_update_oracle;
  constr_comp_model.i_reset = o_constr_comp_model.i_reset;  
  
end