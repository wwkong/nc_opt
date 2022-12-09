%% SPDX-License-Identifier: MIT
% Copyright © 2022 Weiwei "William" Kong

% Set up paths.
run('../../../init.m');

% A collection of QP3 experiments for the DP.ADMM paper.
 iter_limit = 1e5;
 tol = 1e-10;

%% n = 10, variable gamma
n = 10;
out_tbl1 = [];
gamma_arr = [10^0, 10^1, 10^2, 10^3, 10^4, 10^5]; 
for i=1:length(gamma_arr)
  gamma = gamma_arr(i);
  iter_row = run_qp3_experiment(n, gamma, iter_limit, tol);
  tbl_row = [table(n, gamma) iter_row];
  if (i == 1)
    out_tbl1 = tbl_row;
  else
    out_tbl1 = [out_tbl1; tbl_row];
  end
end
disp(out_tbl1);

%% variable n, gamma = 100
gamma = 100;
out_tbl2 = [];
n_arr = [10 * 4^0, 10 * 4^1, 10 * 4^2, 10 * 4^3, 10 * 4^4, 10 * 4^5]; 
for i=1:length(n_arr)
  n = n_arr(i);
  iter_row = run_qp3_experiment(n, gamma, iter_limit, tol);
  tbl_row = [table(n, gamma) iter_row];
  if (i == 1)
    out_tbl2 = tbl_row;
  else
    out_tbl2 = [out_tbl2; tbl_row];
  end
end
disp(out_tbl2);

%% Create a workbook of the results.
writetable(out_tbl1, 'qp3_results.xlsx', 'Sheet', 1);
writetable(out_tbl2, 'qp3_results.xlsx', 'Sheet', 2);