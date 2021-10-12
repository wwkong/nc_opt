% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function x_prox = prox_l1(x, lambda)
% The L1 proximal operator.
  x_prox = sign(x) .*  max(0, abs(x) - lambda); % optimized
end