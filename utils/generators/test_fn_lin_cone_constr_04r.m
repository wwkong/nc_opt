%{

FILE DATA
---------
Last Modified: 
  August 17, 2020
Coders: 
  Weiwei Kong

%}

function [oracle, params] = test_fn_lin_cone_constr_04r(N, r, M, m, seed, dimM, dimN, density)
% Generator of a test suite of unconstrained nonconvex quadratic SDP 
% functions. Data matrices are sparse and their densities are calibrated
% according to the input variable 'density'.
% 
% Note:
% 
%   - xi and tau are chosen so that the curvature pair is (M, m)
%   - Entries of A, B, and C are drawn randomly from a U(0,1) distribution
%   - A and C are dimM-by-dimN-by-dimN sized matrices
%   - b is defined as b = A * (E / dimN) where E = diag(e) and e is a vector 
%     of all ones
%   - D is a diagonal matrix with integer elements from [1, N]
%   - Function is -xi / 2 * ||D * B * Z|| ^ 2 + tau / 2 * ||C * Z - d|| ^ 2 
%   - Gradient is -xi * B' * (D' * D) * B * Z + tau *  C' * (C * Z - d)
%   - Constraint is A(Z) = b
%   - Domain of composite function is the set of matrices with eigenvlaues between 0 and r/sqrt(n).
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
  A = sprand(dimM, dimN * dimN, density);
  d = rand([dimM, 1]);
  
  % Choose (xi, tau).
  [tau, xi, Dfn, Z] = eigen_bisection(M, m, C, D * B);
  
  % Set the topology (Euclidean).
  prod_fn = @(a,b) sum(dot(a, b));
  norm_fn = @(a) norm(a, 'fro');
  
  % Computing norm of A.
  Hp = A * A';
  Hp = (Hp + Hp') / 2;
  norm_A = sqrt(eigs(Hp, 1, 'la')); % same as lamMax(A'*A)
  
  % Set up helper tensors and operators.
  A_tsr = ndSparse(A, [dimM, dimN, dimN]);
  At_tsr = permute(A_tsr, NDIMS:-1:1);
  B_tsr = ndSparse(B, [dimN, dimN, dimN]);
  Bt_tsr = permute(B_tsr, NDIMS:-1:1);
  C_tsr = ndSparse(C, [dimM, dimN, dimN]);
  Ct_tsr = permute(C_tsr, NDIMS:-1:1);  
  lin_op = @(Q, x) full(tsr_mult(Q, x, 'primal'));
  adj_op = @(Qt, y) sparse(tsr_mult(Qt, y, 'dual'));
  
  % Compute the b vector
  E = diag(rand(dimN, 1) * r);
  b = lin_op(A_tsr, E);
  
  % Constraint map methods.
  params.constr_fn = @(Z) lin_op(A_tsr, Z) - b;
  params.grad_constr_fn = @(Z) At_tsr;
  params.set_projector = @(Z) zeros(size(b));
  params.K_constr = norm_A;
  
  % Basic output params.
  params.M = eigs(Dfn(xi, tau) * Z, 1, 'lr');
  params.m = -eigs(Dfn(xi, tau) * Z, 1, 'sr');
  params.x0 = zeros(dimN);
  params.prod_fn = prod_fn;
  params.norm_fn = norm_fn;
  
  % Special params for individual constraints.
  params.K_constr_vec = full(sqrt(sum(A .^ 2, 2)));
  params.L_constr_vec = zeros(dimM, 1);
  params.m_constr_vec = zeros(dimM, 1);

  % Create the Oracle object.
  f_s = @(x) -xi / 2 * norm_fn(D * lin_op(B_tsr, x)) ^ 2 + tau / 2 * norm_fn(lin_op(C_tsr, x) - d) ^ 2;
  f_n = @(x) 0;
  grad_f_s = @(x) -xi * adj_op(Bt_tsr, (D' * D) * lin_op(B_tsr, x)) + tau * adj_op(Ct_tsr, lin_op(C_tsr, x) - d);
  prox_f_n = @(x, lam) box_mat_proj(x, 0, r);  
  oracle = Oracle(f_s, f_n, grad_f_s, prox_f_n);

end