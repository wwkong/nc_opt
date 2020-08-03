%{
 
DESCRIPTION
-----------
Given a point, produce the projection of the point x onto the alpha- 
simplex of dimension length(x).

FILE DATA
---------
Last Modified: 
  August 2, 2020
Coders: 
  Weiwei Kong

INPUT
-----
x:
  The vector or matrix being projected.
alpha:
  A parameter of the simplex.

OUTPUT
------
xP:
  A vector or matrix representing the solution point.

%}

function xP = sm_proj(x, alpha)

y = x / alpha;
n = length(y);
u = sort(y, 'descend');
csU = cumsum(u);
iPos = (u + 1 ./ (1:n)' .* (1 - csU)) > 0;
rho = find(iPos, 1, 'last');
lambda = 1 / rho * (1 - csU(rho));
yP = max(y + lambda, 0);
xP = yP * alpha;

end
