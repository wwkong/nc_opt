% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [spectral_oracle, params] = test_fn_spectral_bmc_01(matrix_arr, alpha, beta, theta, mu, seed)
% Generates the needed functions for the regularized block matrix completion problem. Requires an array of matrices to make up 
% the blocks.
% 
% Note:
% 
%   This version also adds:
% 
%     1. A ball constraint of the form ||X||_F <= R, where R := sqrt(m * n) * ||X||_infinty.
%     2. A kappa_tilde(||x||^2/2) penalty (scaled by alpha) to the f2 component, where kappa_tilde(t) := beta * (1 - exp(-t / theta)).
%
% Arguments:
% 
%   matrix_arr (double vector array): An array of matrices to populate the blocks of the matrices to be completed.
% 
%   beta (double): One of the objective function's hyperparameters.
% 
%   theta (double): One of the objective function's hyperparameters.
% 
%   mu (double): One of the objective function's hyperparameters. 
% 
%   seed (int): The number used to seed MATLAB's random number generator. 
% 
% Returns:
%
%   A pair consisting of a SpectralOracle and a struct. The spectral 
%   oracle is a first-order oracle underyling the optimization problem and 
%   the struct contains the relevant hyperparameters of the problem. 
% 

  rng(seed);
  
  % Parse input
  data = sparse(blkdiag(matrix_arr{:}));
  blk_sizes = cellfun(@size, matrix_arr, 'UniformOutput', false);
  blk_min_dims = cellfun(@min, blk_sizes);
  
  % Initialize
  [rId, cId] = find(data);
  [dim_m, dim_n] = size(data);
  k = nnz(data);
  P = zeros(dim_m, dim_n);
  for i=1:k
      P(rId(i), cId(i)) = 1;
  end

  % Other params
  norm_fn = @(a) norm(a, 'fro');
  rho = beta / theta ^ 2;
  
  % Define R using the bound ||x||_2 <= sqrt(n) ||x||_infinity when x is a
  % vector of dimension n.
  [~, ~, data_vals] = find(data);
  min_data = min(data_vals);
  max_data = max(data_vals);
  range_data = max_data - min_data;
  upper_data = ceil(range_data);
  mean_data = mean(data_vals);
  max_mag_data = max(abs([min_data, max_data]));
  R = sqrt(dim_n * dim_m) * max_mag_data;

  % Output params
  params.norm_fn = norm_fn;
  params.m1 = 0;
  params.m2 = 2 * rho * mu + alpha * 2 * beta / theta * exp(-3 * theta / 2);
  params.M1 = 1;
  params.M2 = alpha * beta / theta;
  params.M = params.M1 + params.M2;
  params.m = params.m1 + params.m2;
  params.x0 = binornd(upper_data, mean_data / upper_data, dim_m, dim_n) - min_data;
  % Projector onto the set Omega
  params.Omega_projection = @(X) ball_projection(X, R); 
  % This function takes in a matrix and returns a triple [P, D, Q] of some
  % spectral decomposition (e.g. SVD, Eigendecomposition, trucated SVD)
  params.decomp_fn = @(X) svd(X, 'econ'); 
   
  % Spectral oracle construction
  function oracle_struct = oracle_eval_fn(X, X_vec)

    % Induced norm
    norm_fn = @(a)   norm(a, 'fro');  

    % Parse inputs
    [dim_m, dim_n] = size(X);
    dim_r = min([dim_m, dim_n]);
    if isempty(X_vec)
      [U, S, V] = svd(X, 'econ');
      s = diag(S);
    else
      U = speye(dim_r);
      V = speye(dim_r);
      s = X_vec;
    end

    % Parse params
    size_data = size(data);
    min_rank = min(size_data);

    % Computed params
    kappa0 = beta / theta;
    mu_bar = mu * kappa0;

    % Basic lambda functions
    kappa = @(t) beta * log(1 + abs(t) / theta);
    kappa_tilde = @(s) beta * (1 - exp(- norm(s) ^ 2 / (2 * theta)));
    kappa_prime = @(t) beta ./ (theta + abs(t)) .* sign(t);
    kappa_tilde_prime = @(s) beta / theta * exp(-norm(s) ^ 2 / (2 * theta)) .* s;
    s_prox_l1 = @(lam) prox_l1(s, lam);
    s_prox_const = @(lam) R / max([R, norm(s_prox_l1(lam), 'fro')]);
    s_prox = @(lam) s_prox_l1(lam) * s_prox_const(lam);

    % Blockwise penalty
    function penalty_val = block_kappa_tilde(s)
      penalty_val = 0;
      idx = 1;
      for i=1:length(blk_min_dims)
        blk_s = s(idx:(idx + blk_min_dims(i) - 1));
        penalty_val = penalty_val + kappa_tilde(blk_s);
        idx = idx + blk_min_dims(i);
      end
    end
    function penalty_prime = block_kappa_tilde_prime(s)
      penalty_prime = zeros(length(s), 1);
      idx = 1;
      for i=1:length(blk_min_dims)
        blk_s = s(idx:(idx + blk_min_dims(i) - 1));
        penalty_prime(idx:(idx + blk_min_dims(i) - 1)) = kappa_tilde_prime(blk_s);
        idx = idx + blk_min_dims(i);
      end
    end

    % Create the (spectral) oracle functions
    oracle_struct.sigma = s;
    oracle_struct.spectral_grad_f2_s = @() alpha * block_kappa_tilde_prime(s) + mu * (kappa_prime(s) - kappa0 * sign(s));
    oracle_struct.spectral_prox_f_n = @(lam) s_prox(lam * mu_bar);

    % Create the (non-spectral) oracle functions
    oracle_struct.f1_s = @() 1/2 * norm_fn(P .* (X - data)) ^ 2;
    oracle_struct.f2_s = @() alpha * block_kappa_tilde(s) + mu * sum(kappa(s) - kappa0 * abs(s));
    oracle_struct.f_n = @() mu_bar * sum(abs(s));
    oracle_struct.grad_f1_s = @() P .* (X - data);
    oracle_struct.grad_f2_s = @() U * bsxfun(@times, oracle_struct.spectral_grad_f2_s(), V');
    oracle_struct.prox_f_n = @(lam) U * bsxfun(@times, oracle_struct.spectral_prox_f_n(lam), V'); 

    % Create the combined functions
    oracle_struct.f_s = @() oracle_struct.f1_s() + oracle_struct.f2_s();
    oracle_struct.grad_f_s = @() oracle_struct.grad_f1_s() + oracle_struct.grad_f2_s();

    % Create the Tier I special oracle constructs
    oracle_struct.f_s_at_prox_f_n = @(lam) ...
      1/2 * norm_fn(P .* (X - data)) ^ 2 + alpha * block_kappa_tilde(s) + mu * sum(kappa(s_prox(lam)) - kappa0 * s_prox(lam)); 
    oracle_struct.f_n_at_prox_f_n = @(lam) mu_bar * sum(s_prox(lam));
    oracle_struct.grad_f_s_at_prox_f_n = @(lam) ...
      P .* (X - data) + U * bsxfun(@times, alpha * block_kappa_tilde_prime(s) + mu * ...
        (kappa_prime(s_prox(lam)) - kappa0 * sign(s_prox(lam))), V(:, 1:min_rank)');
  end
  spectral_oracle = SpectralOracle(@oracle_eval_fn);

end