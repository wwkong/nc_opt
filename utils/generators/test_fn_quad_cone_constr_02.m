%{

FILE DATA
---------
Last Modified: 
  September 13, 2020
Coders: 
  Weiwei Kong

%}

function [oracle, params] = ...
  test_fn_quad_cone_constr_02(N, M, m, seed, dimM, dimN, density)
% Generator of a test suite of unconstrained nonconvex quadratically 
% constrained quadratic SDP functions. Data matrices are sparse and 
% their densities are calibrated according to the input variable 'density'.
% 
% Note:
% 
%   - xi and tau are chosen so that the curvature pair is (M, m)
%   - Entries of B and C are drawn randomly from a U(0,1) distribution
%   - D is a diagonal matrix with integer elements from [1, N]
%   - Function is: -xi / 2 * ||D * B * Z|| ^ 2 + tau / 2 * ||C * Z - d|| ^ 2 
%   - Gradient is: -xi * B' * (D' * D) * B * Z + tau *  C' * (C * Z - d)
%   - Constraint is:
%       (1 / 2) * (P * Z)' * (P * Z) + (Q' * Q * Z) + (Z' * Q' * Q) <= I
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
%   A pair consisting of an Oracle and a struct. The oracle is first-order
%   oracle underyling the optimization problem and the struct contains the
%   relevant hyperparameters of the problem. 
% 
 
  % Initialize.
  rng(seed);
  NDIMS = 3;
  D = sparse(1:dimN, 1:dimN, randi([1, N], 1, dimN), dimN, dimN);
  C = sprand(dimM, dimN * dimN, density);
  B = sprand(dimN, dimN * dimN, density);
  P = rand(dimN, dimN) / sqrt(dimN);
  Q = rand(dimN, dimN) / dimN;
  d = rand([dimM, 1]);
  
  % Choose (xi, tau).
  [tau, xi, Dfn, Z] = eigen_bisection(M, m, C, D * B);
  
  % Set the topology (Euclidean).
  prod_fn = @(a,b) sum(dot(a, b));
  norm_fn = @(a) norm(a, 'fro');
  
  % Computing auxiliary constants of P and Q.
  fro_P = norm(P, 'fro');
  fro_Q = norm(Q, 'fro');
  PtP_vec = reshape(P' * P, dimN * dimN, 1);
  QtQ_vec = reshape(Q' * Q, dimN * dimN, 1);
  
  % Set up helper tensors and operators.
  B_tsr = ndSparse(B, [dimN, dimN, dimN]);
  Bt_tsr = permute(B_tsr, NDIMS:-1:1);
  C_tsr = ndSparse(C, [dimM, dimN, dimN]);
  Ct_tsr = permute(C_tsr, NDIMS:-1:1);  
  lin_op = @(M, x) full(tsr_mult(M, x, 'primal'));
  adj_op = @(Mt, y) sparse(tsr_mult(Mt, y, 'dual'));
  
  % Constraint map methods.
  params.constr_fn = @(Z) ...
    (1 / 2) * (P * Z)' * (P * Z) + ...
    (1 / 2) * (Q' * Q) * Z + (1 / 2) * Z' * (Q' * Q) ...
    - eye(dimN) / (dimN ^ 2);
  params.grad_constr_fn = @(Z, Delta) ...
    (1 / 2) * (P' * P) * Z * Delta + (1 / 2) * (P' * P) * Z * Delta' + ...
    (1 / 2) * (Q' * Q) * Delta + (1 / 2) * (Q' * Q) * Delta';
  params.set_projector = @(Z) psd_cone_proj(Z);
  params.dual_cone_projector = @(Z) psd_cone_proj(Z);
  params.K_constr = fro_P ^ 2 / 2 + fro_Q ^ 2;
  params.L_constr = fro_P ^ 2;
  
  % Basic output params.
  params.M = eigs(Dfn(xi, tau) * Z, 1, 'lr');
  params.m = -eigs(Dfn(xi, tau) * Z, 1, 'sr');
  params.x0 = zeros(dimN, dimN);
  params.prod_fn = prod_fn;
  params.norm_fn = norm_fn;
  
  % Special params for individual constraints.
  params.K_constr_vec = PtP_vec / 2 + QtQ_vec;
  params.L_constr_vec = PtP_vec;
  params.m_constr_vec = zeros(dimN * dimN, 1);

  % Create the Oracle object.
  f_s = @(x) ...
    -xi / 2 * norm_fn(D * lin_op(B_tsr, x)) ^ 2 + ...
    tau / 2 * norm_fn(lin_op(C_tsr, x) - d) ^ 2;
  f_n = @(x) 0;
  grad_f_s = @(x) ...
      -xi * adj_op(Bt_tsr, (D' * D) * lin_op(B_tsr, x)) + ...
      tau * adj_op(Ct_tsr, lin_op(C_tsr, x) - d);
  prox_f_n = @(x, lam) psd_box_proj(x);
  oracle = Oracle(f_s, f_n, grad_f_s, prox_f_n);
  
end

% Projection functions
function XP = psd_cone_proj(X)
  % Projection onto the positive semidefinite cone.
  [Q, d] = eig((X + X') / 2, 'vector'); 
  d_nn = max(d, 0);
  XP = Q * diag(d_nn) * Q';
end

function XP = psd_box_proj(X)
  % Projection onto the set of matrices with eigenvalues between 0 and 
  % 1 / sqrt(n).
  [Q, d] = eig((X + X') / 2, 'vector');
  n = length(d);
  dP = min(max(d, 0), 1 / sqrt(n));
  XP = Q * diag(dP) * Q';
end