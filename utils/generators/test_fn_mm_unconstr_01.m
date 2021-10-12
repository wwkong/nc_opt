% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [oracle_factory, params] = test_fn_mm_unconstr_01(k, N, M, m, seed, dimM, dimN, density)
% Generator of a test suite of minimax unconstrained nonconvex QP functions.
%
% See Also:
%
%   utils/generators/test_fn_unconstr_01
%
% Arguments:
%
%   k (int): The number of quadratic functions to generate.
%
%   N (int): One of the objective function's hyperparameters.
%
%   dimM (int): One of the objective function's hyperparameters.
%
%   dimN (int): One of the objective function's hyperparameters.
% 
%   M (double): The target upper curvature of the objective function.
% 
%   m (double): The target lower curvature of the objective function.
% 
%   seed (int): The number used to seed MATLAB's random number generator. 
% 
% Returns:
%
%   A pair consisting of a function and a struct. The function takes in one argument which, when evaluated at $\xi$, outputs an 
%   Oracle object representing the smoothed primal function with smoothing parameter $\xi$. The struct contains the relevant 
%   hyperparameters of the problem. 
% 

  % Set k-invariant parameters
  rng(seed);
  norm_fn = @(a) norm(a, 'fro'); 
  params.x0 = ones(dimN, 1) / dimN;
  params.m = m;
  params.L_x = M;
  params.D_x = 1;
  params.D_y = 1;
  
  % Initialize
  D_arr = cell(k, 1);
  C_arr = cell(k, 1);
  B_arr = cell(k, 1);
  d_arr = cell(k, 1);
  alpha_arr = cell(k, 1);
  beta_arr = cell(k, 1);

  % Construct the auxillary vectors
  for i=1:k
    D_arr{i} = sparse(1:dimN, 1:dimN, randi([1, N], 1, dimN), dimN, dimN);
    C_arr{i} = sprand(dimM, dimN, density);
    B_arr{i} = sprand(dimN, dimN, density);
    d_arr{i} = rand([dimM, 1]);
    [alpha_arr{i}, beta_arr{i}, ~, ~] = eigen_bisection(M, m, C_arr{i}, D_arr{i} * B_arr{i});
  end
  
  % Compute L_y and max(|grad_f_s|)
  grad_f_s_vec_ord0 = zeros(dimN, k);
  for i=1:k
    grad_f_s_vec_ord0(:, i) = alpha_arr{i} * C_arr{i}' * d_arr{i};
  end
  params.L_y = sqrt(k) * params.L_x + norm(grad_f_s_vec_ord0, 2);
  
  function out_oracle = oracle_factory_fn(xi)    
  % Oracle factory that constructs oracles based on the smoothing 
  % parameter $\xi$.
  
    function oracle_struct = oracle_eval_fn(x)
      % Compute y_xi and the (weighted) function value.
      f_s_eval_arr = zeros(k, 1);
      for i=1:k
        f_s_eval_arr(i) = ...
          -beta_arr{i} / 2 * norm_fn(D_arr{i} * B_arr{i} * x) ^ 2 + alpha_arr{i} / 2 * norm_fn(C_arr{i} * x - d_arr{i}) ^ 2;
      end
      y_xi = sm_proj(xi * f_s_eval_arr, 1);
      oracle_struct.f_s = @() y_xi' * f_s_eval_arr - norm(y_xi) ^ 2 / (2 * xi);

      % Computes the (weighted) gradient.
      grad_f_s_eval_arr = zeros(dimN, 1, k);
      for i=1:k
          grad_f_s_eval_arr(:, i) = y_xi(i) * (-beta_arr{i} * (B_arr{i}' * (D_arr{i}' * D_arr{i}) * B_arr{i} * x) + ...
                                               alpha_arr{i} * (C_arr{i}' * C_arr{i} * x - C_arr{i}' * d_arr{i}));
      end
      oracle_struct.grad_f_s = @() sum(grad_f_s_eval_arr, 3);
      
      % Create the other suboracles.
      oracle_struct.f_n = @() 0;
      oracle_struct.prox_f_n = @(lam) sm_proj(x, 1);
    end

    % Create the oracle
    out_oracle = Oracle(@oracle_eval_fn);
    
  end % End oracle_factory_fn.
  
  % Prepare the output.
  oracle_factory = @oracle_factory_fn;
 
end

