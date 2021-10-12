% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function xP = sm_proj(x, alpha)
% Given a point, produce the projection of the point x onto the alpha-simplex of dimension length(x).

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
