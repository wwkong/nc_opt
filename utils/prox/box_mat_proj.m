% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function XP = box_mat_proj(X, alpha, beta)
% Given a matrix X, produce the projection of the point X onto the set of matrices with eigenvalues between alpha (low) and 
% beta (high).
  [Q, d] = eig((X + X') / 2, 'vector');
  dP = box_proj(d, alpha, beta);
  XP = Q * diag(dP) * Q';
end
