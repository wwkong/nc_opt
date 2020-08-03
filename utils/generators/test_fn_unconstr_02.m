%{

DESCRIPTION
-----------
Generator of a test suite of unconstrained nonconvex quadratic SDP 
functions. Data matrices are sparse and their densities are calibrated
according to the input variable 'density'.

NOTES
-----
** xi and tau are chosen so that upper and lower curvatures are M and -m 
   respectively
** Entries of A, B, and b are drawn randomly from a U(0, 1) distribution
** D is a diagonal matrix with integer elements chosen at random 
  from [1, N]
** Function is -xi / 2 * ||D * B * Z|| ^ 2 + tau / 2 * ||C * Z - d|| ^ 2 
** Gradient is -xi * B' * (D' * D) * B * Z + tau *  C' * (C * Z - d)

FILE DATA
---------
Last Modified: 
  August 2, 2020
Coders: 
  Weiwei Kong

INPUT
-----
(N, M, m, seed, dimM, dimN, density):
  Input parameters as described in the function description.

INPUT
-----
oracle:
  An Oracle object.
params:
  A struct containing input parameters for this function.

%}

function [oracle, params] = ...
  test_fn_unconstr_02(N, M, m, seed, dimM, dimN, density)
 
  % Initialize
  rng(seed);
  D = sparse(1:dimN, 1:dimN, randi([1, N], 1, dimN), dimN, dimN);
  C = sprand(dimM, dimN * dimN, density);
  B = sprand(dimN, dimN * dimN, density);
  d = rand([dimM, 1]);
  [tau, xi, Dfn, Z] = eigen_bisection(M, m, C, D * B);
  
  % Set the topology (Euclidean)
  prod_fn = @(a,b) sum(dot(a, b));
  norm_fn = @(a) norm(a, 'fro');
  
  % Output params
  params.M = eigs(Dfn(xi, tau) * Z, 1, 'lr');
  params.m = -eigs(Dfn(xi, tau) * Z, 1, 'sr');
  params.x0 = diag(ones(dimN, 1) / dimN);
  params.prod_fn = prod_fn;
  params.norm_fn = norm_fn;

  % Create the Oracle object
  f_s = @(x) ...
    -xi / 2 * norm_fn(D * lin_op(B, x)) ^ 2 + ...
    tau / 2 * norm_fn(lin_op(C, x) - d) ^ 2;
  f_n = @(x) 0;
  grad_f_s = @(x) ...
    -xi * adj_op(B, (D' * D) * lin_op(B, x)) + ...
    tau * (adj_op(C, lin_op(C, x)) - adj_op(C, d));
  prox_f_n = @(x, lam) sm_mat_proj(x, 1);
  oracle = Oracle(f_s, f_n, grad_f_s, prox_f_n);

end

%% Efficient operators used for A(X)

% Function that applies the operator A(:,:,:) to Z(:,:) and returns a
% vector that is the size of the third index of A
function AZ = lin_op(A, Z)
  [n, ~] = size(Z);
  tildeZ = reshape(Z, [n * n, 1]);
  AZ = A * tildeZ;
end
% Function that applies the adjoint of the operator A(:,:,:) to y(:) 
% and returns a matrix
function adj_Ay = adj_op(A, y)
  [~, sqrN] = size(A);
  n = sqrt(sqrN);
  ytildeA = diag(sparse(y)) * A;
  rsum_ytildeA = sum(ytildeA, 1);
  adj_Ay = reshape(rsum_ytildeA, [n, n]);
end