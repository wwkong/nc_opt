% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [tau, xi, Dfn, Z] = eigen_bisection(M, m, A, B)
% Finds nonnegative constants tau and xi such that the matrix C := tau * A' * A - xi * B' * B has maximum eigenvalue M and minimum 
% eigenvalue m.
%
% Arguments:
%  
%   M (double): target maximum eigenvalue
%
%   m (double): target minimum eigenvalue
%
%   A (double): positive matrix component of C
%
%   B (double): negative matrix component of C
%
% Returns:
%
%   Scalars and matrices related to the solution system.
%

  % Initialize
  [mA, nA] = size(A);
  tol = 1e-10;
  ratio_true = M / m;
  tau = 1;
  xi  = 1e-6;
  iter = 0;
  Dfn = 0;
  Z = 0;
  
  % Check if input matrices are dense
  iDense = (~issparse(A) | ~issparse(B));
  
  % Create an oracle to get lamMax(X),lamMin(X) as a function of tau, xi,
  % and the dimensions of Hp,Hn
  if (mA >= nA || iDense)
    Hn = B' * B;
    Hp = A' * A;
    lam_max = @(xi, tau) eigs(-xi * Hn + tau * Hp, 1, 'la');
    lam_min = @(xi, tau) eigs(-xi * Hn + tau * Hp, 1, 'sa');
  else % Low rank approximation approach
    [Dfn,Z] = low_rank_factor(A, B);
    lam_max = @(xi, tau) eigs(Dfn(xi, tau) * Z, 1, 'lr');
    lam_min = @(xi, tau) eigs(Dfn(xi, tau) * Z, 1, 'sr');
  end

  % Find the boundaries
  diff_err = 0; 
  % Upper
  while (diff_err <= 0)
    tau = tau*2;
    ratio_cur = abs(lam_max(xi, tau) / lam_min(xi, tau));
    diff_err = ratio_cur - ratio_true;
    iter = iter + 1;
  end
  tauU = tau;
  % Lower
  while (diff_err > 0)
    tau = tau / 2;
    ratio_cur = abs(lam_max(xi, tau) / lam_min(xi, tau));
    diff_err = ratio_cur - ratio_true;
    iter = iter + 1;
  end
  tauL = tau;
  
  % Closed interval bisection method
  relErr = Inf;
  while (relErr > tol)
    tauM = (tauU + tauL) / 2;
    ratio_cur = abs(lam_max(xi, tauM) / lam_min(xi, tauM));
    if (ratio_cur > ratio_true)
      tauU = tauM;
    else
      tauL = tauM;
    end
    relErr = abs((ratio_cur - ratio_true) / ratio_true);
    iter = iter + 1;
  end
  tau = tauM;
  lam_max_F = lam_max(xi, tau);
  lam_min_F = lam_min(xi, tau);
  
  % Normalize
  mult = min(abs(M / lam_max_F), abs(m / lam_min_F));
  xi = xi * mult;
  tau = tau * mult;
  
end

function [Dfn, Z] = low_rank_factor(A, B)
% Given C(xi, tau) := tau * A' * A - xi * B' * B = C, find diagonal matrix D := D(xi, tau) and matrix Z such that 
% lambda{X} = lambda{D * Z}. This function should only be called if A, B is in R^(m*n) and n >> m.

  % Check if we need this subroutine
  [mA, nA] = size(A);
  [mB, nB] = size(B);
  if (mA > nA || mB > nB)
    error('At * A is already small!');
  end
 
  % Form SVD of A and B, as well as sparse constructor matrices / vectors
  [~, SA, VA] = svds(A, mA);
  [~, SB, VB] = svds(B, mB);
  V = [sparse(VA), sparse(VB)];
  SAvec = diag(SA);
  SBvec = diag(SB);
  
  % Form Dfn and Z matrix
  Dfn = @(xi, tau) sparse(diag([tau * SAvec.^2; -xi * SBvec .^ 2]));
  Z = V' * V;

end