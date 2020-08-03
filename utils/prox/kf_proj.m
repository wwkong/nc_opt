%{

DESCRIPTION
-----------
Given a matrix X and an integer k > 0, produce the projection of the point 
x onto the k-Fantope

FILE DATA
---------
Last Modified: 
  August 2, 2020
Coders: 
  Weiwei Kong

Inputs 
-------
X:
  The vector or matrix being projected.
k:
  A parameter of the Fantope.

Outputs 
-------
XP:
  A vector or matrix representing the solution point.

%}

function XP = kf_proj(X, k)

  % Compute with the symmetric part to smooth out rounding errors
  % i.e. ensure the eigendecomposition produces only real eigenvalues
  [Q, d] = eig((X + X')/2, 'vector');
  dP = kf_sub(d, k);
  XP_n = Q * diag(dP) * Q';
  XP = (XP_n + XP_n') / 2;
  
end

% Solves the Fantope k-dimensional subproblem of finding theta such that 
% sum(min(max(x - theta, 0), 1)) = k
function xKF = kf_sub(x, k)

  % Safety checks
  if (max(size(x)) < k)
    error('Incompatible dimensions n and k!')
  end
  
  % Main code
  m = min(x)-1;
  M = max(x);
  theta = (m + M) / 2;
  xKF_fn = @(theta) min(max(x - theta, 0), 1);
  sum_normalized = @(theta) sum(xKF_fn(theta));
  tol = 1e-10;
  incumbent = sum_normalized(theta);
  while (abs(incumbent - k) > tol)
    if (incumbent > k)
      m = theta;
    else
      M = theta;
    end
    theta = (m + M) / 2;
    incumbent = sum_normalized(theta);
  end
  xKF = xKF_fn(theta);
  
end