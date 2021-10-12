% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [oracle, params] = test_fn_unconstr_02(N, M, m, seed, dimM, dimN, density)
% Generator of a test suite of unconstrained nonconvex quadratic SDP functions. Data matrices are sparse and their densities are
% calibrated according to the input variable 'density'.
%
% Arguments:
%  
%   N (int): One of the objective function's hyperparameters.
%
%   dimM (int): One of the objective function's hyperparameters.
%
%   dimN (int): One of the objective function's hyperparameters.
%
%   density (double): The density level of the generated matrices.
% 
%   M (double): The target upper curvature of the objective function.
% 
%   m (double): The target lower curvature of the objective function.
% 
%   seed (int): The number used to seed MATLAB's random number generator. 
% 
% Returns:
%
%   A pair consisting of an Oracle and a struct. The oracle is first-order oracle underyling the optimization problem and the 
%   struct contains the relevant hyperparameters of the problem. 
% 
 
  % Initialize.
  rng(seed);
  NDIMS = 3;
  D = sparse(1:dimN, 1:dimN, randi([1, N], 1, dimN), dimN, dimN);
  C = sprand(dimM, dimN * dimN, density);
  B = sprand(dimN, dimN * dimN, density);
  d = rand([dimM, 1]);
  [tau, xi, Dfn, Z] = eigen_bisection(M, m, C, D * B);
  
  % Set the topology (Euclidean).
  prod_fn = @(a,b) sum(dot(a, b));
  norm_fn = @(a) norm(a, 'fro');
  
  % Output params.
  params.M = eigs(Dfn(xi, tau) * Z, 1, 'lr');
  params.m = -eigs(Dfn(xi, tau) * Z, 1, 'sr');
  params.x0 = eye(dimN) / dimN;
  params.prod_fn = prod_fn;
  params.norm_fn = norm_fn;
  
  % Set up helper tensors and operators.
  B_tsr = ndSparse(B, [dimN, dimN, dimN]);
  Bt_tsr = permute(B_tsr, NDIMS:-1:1);
  C_tsr = ndSparse(C, [dimM, dimN, dimN]);
  Ct_tsr = permute(C_tsr, NDIMS:-1:1);
  lin_op = @(Q, x) full(tsr_mult(Q, x, 'primal'));
  adj_op = @(Qt, y) sparse(tsr_mult(Qt, y, 'dual'));

  % Create the Oracle object.
  f_s = @(x) -xi / 2 * norm_fn(D * lin_op(B_tsr, x)) ^ 2 + tau / 2 * norm_fn(lin_op(C_tsr, x) - d) ^ 2;
  f_n = @(x) 0;
  grad_f_s = @(x) -xi * adj_op(Bt_tsr, (D' * D) * lin_op(B_tsr, x)) + tau * adj_op(Ct_tsr, lin_op(C_tsr, x) - d);
  prox_f_n = @(x, lam) sm_mat_proj(x, 1);
  oracle = Oracle(f_s, f_n, grad_f_s, prox_f_n);

end