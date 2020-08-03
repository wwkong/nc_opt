%{

DESCRIPTION
-----------
The L1 proximal point operator.

FILE DATA
---------
Last Modified: 
  August 2, 2020
Coders: 
  Weiwei Kong

INPUT
-----
x:
  The vector or matrix being evaluated.
lambda:
  The prox scaling parameter.

OUTPUT
------
x_prox:
  A vector or matrix representing the solution point.

%}

function x_prox = prox_l1(x, lambda)
  % prox mapping of x wrt (lambda * l1)-norm
  x_prox = sign(x) .*  max(0, abs(x) - lambda); % optimized
end