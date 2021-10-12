% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function xP = box_proj(x, alpha, beta)
% Given a point, produce the projection of the point x onto the set of vectors with entries between alpha (low) and beta (high).
  xP = min(max(x, alpha), beta);
end
