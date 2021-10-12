% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function proj = normal_eigenbox_proj(X, Z, r)
% Projects Z onto the normal cone of the set of positive semidefinite matrices whose eigenvalues are bounded above by r at Z.

  % Set up helper variables.
  tol = 1e-12;
  [Q, D] = eig((X + X')/2);
  d = diag(D);
  lo_mask = abs(d) < tol;
  hi_mask = abs(r - d) < tol;
  Q_lo = Q(:, lo_mask);
  Q_hi = Q(:, hi_mask);
  
  % Main computation (do NOT assume that Z is symmetric)!
  Zs = (Z' + Z) / 2;
  Y_lo = full(Q_lo' * Zs * Q_lo);
  [P_lo, D_lo] = eig((Y_lo + Y_lo') / 2);
  C_lo = P_lo * min(0, D_lo) * P_lo';
  Y_hi = full(Q_hi' * Zs * Q_hi);
  [P_hi, D_hi] = eig((Y_hi + Y_hi') / 2);
  C_hi = P_hi * max(0, D_hi) * P_hi';
  proj = Q_lo * C_lo * Q_lo' + Q_hi * C_hi * Q_hi';
  proj = (proj' + proj) / 2;

end