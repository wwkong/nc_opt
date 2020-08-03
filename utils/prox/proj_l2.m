%{

DESCRIPTION
-----------
Projects a point onto the Euclidean ball of radius 'a'.

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
a:
  The radius of the Euclidean ball.

OUTPUT
------
y:
  The projected point.

%}

function y = proj_l2(x, a)
  if norm(x) > a
      y = x / norm(x) * a;
  else
      y = x;
  end
end