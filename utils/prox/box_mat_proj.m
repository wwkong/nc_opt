%{

DESCRIPTION
-----------
Given a matrix X, produce the projection of the point X onto the set of matrices with eigenvalues between alpha (low) and 
beta (high).

FILE DATA
---------
Last Modified: 
  August 2, 2020
Coders: 
  Weiwei Kong

INPUT 
-----
X:
  The vector or matrix being projected.
alpha:
  A parameter of the spectral box.

OUTPUT 
------
XP:
  A vector or matrix representing the solution point.

%}

function XP = box_mat_proj(X, alpha, beta)

  % Compute with the symmetric part to smooth out rounding errors
    % i.e. ensure the eigendecomposition produces only real eigenvalues
  [Q, d] = eig((X + X') / 2, 'vector');
  dP = box_proj(d, alpha, beta);
  XP = Q * diag(dP) * Q';

end
