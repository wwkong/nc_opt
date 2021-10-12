% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function XP = sm_mat_proj(X, alpha)
% Given a matrix X, produce the projection of the point X onto the alpha-spectrahedron of dimension size(X).

  % Compute with the symmetric part to smooth out rounding errors
    % i.e. ensure the eigendecomposition produces only real eigenvalues
  [Q, d] = eig((X + X') / 2, 'vector');
  dP = sm_proj(d, alpha);
  XP = Q * diag(dP) * Q';

end
