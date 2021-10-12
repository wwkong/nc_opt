% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [xi, M] = compute_smoothed_parameters(params, type)
% Computes the smoothing parameter of a minmax problem based on the curvatures, tolerances, and algorithm.
  
  % Parse inputs.
  rho_y = params.rho_y;
  L_y = params.L_y;
  L_x = params.L_x;
  m = params.m;
  
  % Compute the relevant parts of the smoothed function.
  xi = params.D_y / rho_y;
  if strcmp(type, 'PGSF')
    M = xi * params.L_y ^ 2 + params.L_x;
  elseif strcmp(type, 'AIPP-S')
    M = L_y * (xi * L_y + sqrt(xi * (L_x + m)));
  else
    error('Unknown smoothing type!');
  end
  
end