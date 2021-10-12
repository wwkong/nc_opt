% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function x_proj = ball_projection(x, R)
% Efficient projection onto the Euclidean ball of radius R.
  size_x = norm(x, 'fro');
  if (size_x <= R)
    x_proj = x;
  else
    x_proj = x / size_x * R;
  end
end