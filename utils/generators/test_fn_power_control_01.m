% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [oracle_factory, params] = test_fn_power_control_01(dim_K, dim_N, sigma, seed)
% Generator of a test suite of nonconvex power control problems.
%
% Arguments:
%
%   dim_K (int): One of the objective function's hyperparameters.
%
%   dim_N (int): One of the objective function's hyperparameters.
%
%   sigma (int): One of the objective function's hyperparameters.
% 
%   seed (int): The number used to seed MATLAB's random number generator. 
% 
% Returns:
%
%   A pair consisting of a function and a struct. The function takes in one argument which, when evaluated at $\xi$, outputs an 
%   Oracle object representing the smoothed primal function with smoothing parameter $\xi$. The struct contains the relevant 
%   hyperparameters of the problem. 
% 

  % Parse input and initialize matrices
  rng(seed);
  H = randn(dim_K * dim_N, dim_K) + 1i * randn(dim_K * dim_N, dim_K);
  P = randn(dim_K, dim_N) + 1i * randn(dim_K, dim_N);
  A = sqrt(real(H .* conj(H)));
  B = sqrt(real(P .* conj(P)));
  R = dim_K ^ (1 / dim_K);
  
  % Compute the array of partial diagonal elements A_{k,k,n}.
  exp_idx = sub2ind(size(A), (1:(dim_N * dim_K)), repmat(1:dim_K, [1, dim_N]));
  A_kkn_reduced = reshape(A(exp_idx), [dim_K, dim_N]);

  % Compute the two arrays with elements sum_{j} A_{k,j,n} and
  % sum_{j} B_{j,n} * A_{k,j,n}, respectively.
  mask_L_x = kron(speye(dim_N), ones(1, dim_K));
  base_arr_for_L_x = (A .^ 2)' * mask_L_x';
  B_cell = mat2cell(B, dim_K, ones(dim_N, 1));
  mask_L_y = blkdiag(B_cell{:})';
  base_arr_for_L_y = A' * mask_L_y';

  % Set invariant parameters
  L_mult = 2 / min(sigma ^ 4, sigma ^ 6);
  params.x0 = zeros(dim_K, dim_N);
  params.L_x = L_mult * max(base_arr_for_L_x, [], 'all');
  params.L_y = L_mult * max(base_arr_for_L_y, [], 'all');
  params.D_y = norm(ones(dim_N)) * dim_N / 2;
  params.m = params.L_x;
  
  function out_oracle = oracle_factory_fn(xi)
    % Oracle factory that constructs oracles based on the smoothing 
    % parameter $\xi$.
    
    function oracle_struct = oracle_eval_fn(X)
      % Compute the partial block A ~* X.
      X_vec = reshape(X, [dim_K * dim_N, 1]);
      A_times_X_blk = bsxfun(@times, A, X_vec);

      % Compute the array of X partial sums, 
      % i.e. S1_{k,n} = sum_{j} A_{j,k,n} * X_{k,n}.
      mask_S1 = kron(speye(dim_N), ones(1, dim_K));
      S1_kn_reduced = A_times_X_blk' * mask_S1';

      % Helper functions.
      S12_y_full_kn = @(y) sigma ^ 2 + bsxfun(@times, B', y)' + S1_kn_reduced;
      S12_y_partial_kn = @(y) S12_y_full_kn(y) - A_kkn_reduced .* X;
      y_ls_fn = @(y) sum(B ./ (S12_y_full_kn(y) .* S12_y_partial_kn(y)), 1)' - y / xi;

      % Compute y_xi by line search.
      low_y = zeros(dim_N, 1);
      high_y = dim_N / 2 * ones(dim_N, 1);
      ls_opt_err = norm(low_y - high_y);
      ls_tol = 1e-9;
      while(ls_opt_err > ls_tol)
        % Eval
        mid_y = (low_y + high_y) / 2;
        mid_ls_val = y_ls_fn(mid_y);
        % Update
        i_pos_mid_ls_val = mid_ls_val > 0;
        i_neg_mid_ls_val = 1 - i_pos_mid_ls_val;
        high_y = mid_y .* i_neg_mid_ls_val + high_y .* (1 - i_neg_mid_ls_val);
        low_y = mid_y .* i_pos_mid_ls_val + low_y .* (1 - i_pos_mid_ls_val);
        ls_opt_err = norm(low_y - high_y);
      end
      y_xi = mid_y;

      % Compute the array with elements 
      % sum_{j} A_{k,j,n} / [S_{j,n} * S_{j,n}^-]
      S_jn_full = S12_y_full_kn(y_xi);
      S_jn_partial = S12_y_partial_kn(y_xi);
      S_prod_cell = mat2cell(1 ./ (S_jn_full .* S_jn_partial), dim_K, ones(dim_N, 1));
      mask_grad_f_s = blkdiag(S_prod_cell{:})';
      A_div_S_prod_sum = A' * mask_grad_f_s';

      % Set up the suboracles.
      oracle_struct.f_s = @() -sum(log(1 + A_kkn_reduced .* X ./ S_jn_partial), 'all') - norm(y_xi, 'fro') ^ 2 / (2 * xi);
      oracle_struct.grad_f_s = @() -A_kkn_reduced ./ S_jn_full + A_div_S_prod_sum - A_kkn_reduced ./ (S_jn_full .* S_jn_partial);
      oracle_struct.f_n = @() 0;
      oracle_struct.prox_f_n = @(lam) min(max(0, X), R);
      
    end % End oracle_eval_fn
    
    % Create the output oracle.
    out_oracle = Oracle(@oracle_eval_fn);
  
  end % End oracle_factory_fn.

  % Prepare the output.
  oracle_factory = @oracle_factory_fn;
 
end