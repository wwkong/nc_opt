%{

DESCRIPTION
-----------
Given a matrix X, produce the projection of the point X onto the 
alpha-spectrahedron of dimension size(X).

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
  A parameter of the spectrahedron.

OUTPUT 
------
XP:
  A vector or matrix representing the solution point.

%}

function XP = sm_mat_proj(X, alpha)

  % Compute with the symmetric part to smooth out rounding errors
    % i.e. ensure the eigendecomposition produces only real eigenvalues
  [Q, d] = eig((X + X') / 2, 'vector');
  dP = sm_proj(d, alpha);
  XP = Q * diag(dP) * Q';

end
