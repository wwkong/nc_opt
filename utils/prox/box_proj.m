%{
 
DESCRIPTION
-----------
Given a point, produce the projection of the point x onto the set of vectors with entries between alpha (low) and beta (high).

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

function xP = box_proj(x, alpha, beta)
  xP = min(max(x, alpha), beta);
end
