% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [oracle, params] = test_fn_quad_cone_constr_02r(N, r, M, m, seed, dimM, dimN, density)
% Generator of a test suite of unconstrained nonconvex quadratically constrained quadratic SDP functions. Data matrices are 
% sparse and their densities are calibrated according to the input variable 'density'.
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
  P = log(M / m) * rand(dimN, dimN) / sqrt(100 * dimN * r);
  Q = rand(dimN, dimN) / dimN;
  d = rand([dimM, 1]);
  
  % Auxiliary matrices
  PtP = P' * P;
  QtQ = Q' * Q;
  
  % Choose (xi, tau).
  [tau, xi, Dfn, Z] = eigen_bisection(M, m, C, D * B);
  
  % Set the topology (Euclidean).
  prod_fn = @(a,b) sum(dot(a, b));
  norm_fn = @(a) norm(a, 'fro');
  
  % Computing auxiliary constants of P and Q.
  PtP_vec = reshape(PtP, dimN * dimN, 1);
  QtQ_vec = reshape(QtQ, dimN * dimN, 1);
  
  % Set up helper tensors and operators.
  B_tsr = ndSparse(B, [dimN, dimN, dimN]);
  Bt_tsr = permute(B_tsr, NDIMS:-1:1);
  C_tsr = ndSparse(C, [dimM, dimN, dimN]);
  Ct_tsr = permute(C_tsr, NDIMS:-1:1);  
  lin_op = @(M, x) full(tsr_mult(M, x, 'primal'));
  adj_op = @(Mt, y) sparse(tsr_mult(Mt, y, 'dual'));
  
  % Constraint map methods.
  params.constr_fn = @(Z) (1 / 2) * Z' * PtP * Z + (1 / 2) * (QtQ * Z + Z' * QtQ) - eye(dimN);
  
  % MONTEIRO (gradient).
  params.grad_constr_fn = @(Z, Delta) (1 / 2) * (PtP * Z * Delta + Delta' * Z' * PtP') + (1 / 2) * (QtQ * Delta + Delta' * QtQ');
  
%   % KONG (gradient).
%   params.grad_constr_fn = @(Z, Delta) + (1 / 2) * PtP * Z * (Delta + Delta') + (1 / 2) * QtQ * (Delta + Delta');
  
  % Basic output params.
  params.M = eigs(Dfn(xi, tau) * Z, 1, 'lr');
  params.m = -eigs(Dfn(xi, tau) * Z, 1, 'sr');
  params.x0 = zeros(dimN, dimN);
  params.prod_fn = prod_fn;
  params.norm_fn = norm_fn;
  
  % Note that |Z| <= r * |ones(dimN,1)| when 0 <= Z <= r * I
  
  % Special params for individual constraints.
  params.K_constr_vec = abs(PtP_vec) / 2 * norm(ones(dimN, 1) * r) + abs(QtQ_vec);
  params.L_constr_vec = abs(PtP_vec);
  params.m_constr_vec = zeros(dimN * dimN, 1);
  
  % Other maps and constants.
  params.set_projector = @(Y) box_mat_proj(Y, 0, Inf);
  params.dual_cone_projector = @(Y) box_mat_proj(Y, 0, Inf);
  params.K_constr = norm(params.K_constr_vec);
  params.L_constr = norm(params.L_constr_vec);

  % Create the Oracle object.
  f_s = @(x) -xi / 2 * norm_fn(D * lin_op(B_tsr, x)) ^ 2 + tau / 2 * norm_fn(lin_op(C_tsr, x) - d) ^ 2;
  f_n = @(x) 0;
  grad_f_s = @(x) -xi * adj_op(Bt_tsr, (D' * D) * lin_op(B_tsr, x)) + tau * adj_op(Ct_tsr, lin_op(C_tsr, x) - d);
  prox_f_n = @(x, lam) box_mat_proj(x, 0, r); 
  oracle = Oracle(f_s, f_n, grad_f_s, prox_f_n);
  
end