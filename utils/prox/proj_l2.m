% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function y = proj_l2(x, r)
% Projects a point onto the Euclidean ball of radius 'a'.
  if norm(x) > r
      y = x / norm(x) * r;
  else
      y = x;
  end
end