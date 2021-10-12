% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [oracle_factory, params] = test_fn_robust_regression_01(data_name, alpha)
% Generator of a test suite of robust regression functions.
%
% Arguments:
%
%   data_name (character vector): Name of the input data (in CSV format).
%
%   alpha (double): One of the objective function's hyperparameters.
% 
% Returns:
%
%   A pair consisting of a function and a struct. The function takes in one argument which, when evaluated at $\xi$, outputs an 
%   Oracle object representing the smoothed primal function with smoothing parameter $\xi$. The struct contains the relevant 
%   hyperparameters of the problem. 
% 

  % Read in data
  [sp_labels, sp_features] = libsvmread(data_name);
  
  % Parse data
  labels = full(sp_labels);
  features = full(sp_features);
  [~, dim_m] = size(features);
  row_norms = sqrt(sum(features .^ 2, 2));

  % Set invariant parameters
  params.x0 = zeros(dim_m, 1);
  params.D_y = 1;
  params.L_y = norm(features, 'fro');
  params.L_x = (max(row_norms) .^ 2) ./ alpha;
  params.m = params.L_x;

  % Initialize
  constr_struct.labels = labels;
  constr_struct.features = features;
  constr_struct.alpha = alpha;
  
  function out_oracle = oracle_factory_fn(xi)    
  % Oracle factory that constructs oracles based on the smoothing 
  % parameter $\xi$.
  
    function oracle_struct = oracle_eval_fn(x)
      % Parse params
      labels = constr_struct.labels;
      features = constr_struct.features;
      alpha = constr_struct.alpha;

      % Set up variables for computing function value
      gap = labels .* features * x;
      i_pos_gap = gap > 0;
      i_neg_gap = 1 - i_pos_gap;
      ell_at_x_arr = i_pos_gap .* log(1 + exp(-gap .* i_pos_gap)) + i_neg_gap .* (-gap + log(1 + exp(gap .* i_neg_gap)));
      phi_alpha_ell_arr = alpha * log(1 + ell_at_x_arr / alpha);

      % Set up variables for computing  gradient value
      logistic_at_x_arr = ...
        i_pos_gap .* (exp(-gap .* i_pos_gap) ./ (1 + exp(-gap .* i_pos_gap))) + i_neg_gap .* (1 ./ (1 + exp(gap .* i_neg_gap)));
      tau_at_x_arr = logistic_at_x_arr ./ (alpha + ell_at_x_arr);

      % Compute y_xi
      y_xi = sm_proj(xi * phi_alpha_ell_arr, 1);

      % Create the oracle
      oracle_struct.f_s = @() y_xi' * phi_alpha_ell_arr - norm(y_xi) ^ 2 / (2 * xi);
      oracle_struct.f_n = @() 0;
      oracle_struct.grad_f_s = @() -alpha * features' * (tau_at_x_arr .* y_xi .* labels);
      oracle_struct.prox_f_n = @(lam) x;
    end % End oracle_eval_fn.
    
    % Create the oracle
    out_oracle = Oracle(@oracle_eval_fn);
    
  end % End oracle_factory_fn.
  
  % Prepare the output.
  oracle_factory = @oracle_factory_fn;
 
end