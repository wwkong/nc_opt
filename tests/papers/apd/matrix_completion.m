% SPDX-License-Identifier: MIT
% Copyright © 2022 Weiwei "William" Kong

% The function of interest is
%
%  f(z) :=  (1 / k) * sum_{i=1,..,k} (1 - tanh(v_i * <u_i, z>)) + (1 / 2*k) ||z|| ^ 2.

%% Initialization

% Set up paths.
run('../../../init.m');

% Set up variables
seed = 777;
alpha = 1E-8;
beta = 500.0;
theta = 1E-4;
mu = 1.0;
global_tol = 1e-6;
time_limit = 1200;
iter_limit = 1000;

%% Generate Tables

% Set hyperparameters.
base_hparam = struct();
base_hparam.i_logging = true;

ncf_hparam = base_hparam;
ncf_hparam.m0 = 1;
ncf_hparam.M0 = 1;

upf_hparam = base_hparam;
upf_hparam.line_search_type = 'monotonic';

apd_hparam = base_hparam;
apd_hparam.m0 = 1;
apd_hparam.M0 = 1;
apd_hparam.theta = 4;
apd_hparam.beta = 2;
apd_hparam.alpha = 2;
apd1_hparam = apd_hparam;
apd1_hparam.line_search_type = 'monotonic';
apd2_hparam = apd_hparam;
apd2_hparam.line_search_type = 'optimistic';

aipp_hparam = base_hparam;
aipp_hparam.steptype = 'aipp';
aipp_hparam.acg_steptype = 'constant';
aipp_hparam.sigma = 1/4;

% Loop over the curvature pair (m, M).
img_names = {'moose.jpg'};
[nrows, ncols] = size(img_names);
offset = 0;

for i = 1:(nrows-offset)
  % Use a problem instance generator to create the oracle and
  % hyperparameters.
  rng(seed);
  snr = 200;
  density = 0.25;
  img_matrix = corrupt_im(imread(img_names{i}), snr, density);
  [oracle, hparams] = test_fn_penalty_mc_01(img_matrix, alpha, beta, theta, mu, seed);

  % Create the Model object and specify the solver.
  mat_compl = CompModel(oracle);

  % Set the curvatures and the starting point x0.
  mat_compl.M = hparams.M;
  mat_compl.m = hparams.m;
  mat_compl.x0 = hparams.x0;
  
  % Adjustments for AIPP.
  aipp_hparam.m = hparams.m;
  aipp_hparam.M = hparams.M;

  % Set up the termination criterion.
  mat_compl.opt_type = 'relative';
  mat_compl.opt_tol = global_tol;
  mat_compl.time_limit = time_limit;
  mat_compl.iter_limit = iter_limit;

  % Run a benchmark test and print the summary.
  solver_arr = {@UPFAG, @ADAP_FISTA, @AIPP, @APD};
  hparam_arr = {upf_hparam, ncf_hparam, aipp_hparam, apd2_hparam};
  name_arr = {'UPF', 'ANCF', 'AIPP', 'APD'};

%   solver_arr = {@ADAP_FISTA, @APD};
%   hparam_arr = {ncf_hparam, apd1_hparam};
%   name_arr = {'ANCF', 'APD1'};
  
%   solver_arr = {@APD};
%   hparam_arr = {apd2_hparam};
%   name_arr = {'APD2'};
  
  [summary_tables, comp_models] = run_CM_benchmark(mat_compl, solver_arr, hparam_arr, name_arr);
  disp(summary_tables.all);
  writetable(summary_tables.all, "adp_mat_compl" + num2str(i + offset) + ".xlsx");
  
  % Set up the final table.
  if (i == 1)
    final_table = summary_tables.all;
  else
    final_table = [final_table; summary_tables.all];
  end
end

% Display final table for logging.
disp(final_table);
writetable(final_table, "adp_mat_compl.xlsx");

%% 
nplots = length(name_arr) + 2;
subplot(1, nplots, 1);
imshow(imread(img_names{i}))
subplot(1, nplots, 2);
imshow(uint8(img_matrix));
for j=1:length(name_arr)
  im_diff = imread(img_names{i}) - uint8(comp_models.(name_arr{j}).model.x);
  disp(norm(double(im_diff), 'fro'))
  subplot(1, nplots, 2 + j);
  imshow(uint8(comp_models.(name_arr{i}).model.x));
end 

%% Generate Plot data.
% 
% m = 1E2;
% M = 1E6;
% 
% upf_hparam.i_logging = true;
% ncf_hparam.i_logging = true;
% aipp_hparam.i_logging = true;
% apd2_hparam.i_logging = true;
% 
% [oracle, hparams] =  test_fn_unconstr_02(N, M, m, seed, dimM, dimN, density);
% ncvx_svm = CompModel(oracle);
% ncvx_svm.M = hparams.M;
% ncvx_svm.m = hparams.m;
% ncvx_svm.x0 = hparams.x0;
% 
% aipp_hparam.m = hparams.m;
% aipp_hparam.M = hparams.M;
% 
% ncvx_svm.opt_type = 'relative';
% ncvx_svm.opt_tol = 1E-6;
% ncvx_svm.time_limit = time_limit;
% ncvx_svm.iter_limit = 20000;
% 
% solver_arr = {@UPFAG, @ADAP_FISTA, @AIPP, @APD};
% hparam_arr = {upf_hparam, ncf_hparam, aipp_hparam, apd2_hparam};
% name_arr = {'UPF', 'ANCF', 'AIPP', 'APD'};
% 
% [summary_tables, comp_models] = run_CM_benchmark(ncvx_svm, solver_arr, hparam_arr, name_arr);
% 
% %% Generate Plots
% oracle.eval(hparams.x0);
% tol_factor = 1 + hparams.norm_fn(oracle.grad_f_s());
% line_styles = {'-.', '--', ':', '-'};
% figure;
% for i=1:max(size(name_arr))
%   semilogy(cummin(comp_models.(name_arr{i}).history.stationarity_values) / tol_factor, ...
%            line_styles{i}, 'LineWidth', 1.5);
%   hold on;
% end
% title("QSDP Residuals vs. Iterations", 'Interpreter', 'latex');
% xlim([1, 1E4]);
% xlabel("Iteration Count", 'Interpreter', 'latex');
% ylim([1E-6, 1E-2]);
% ylabel("$$\min_{1\leq i \leq k} \|\bar{v}_i\| / (1 + \|\nabla f(z_0)\|)$$", 'Interpreter', 'latex');
% legend(name_arr);
% ax = gca;
% ax.FontSize = 16;
% hold off;
% saveas(gcf,'qsdp.svg')

%% Helper functions.

function I2 = corrupt_im(I1, snr, density)
  I2 = add_random_zeros(double(add_noise(double(I1), snr)), density);
end

function err = compute_l2_err(I1, I2)
  im_diff = uint8(I1) - uint8(I2);
  err = norm(double(im_diff), 'fro');
end

function tbl = get_l2_err_tbl(comp_models, name_arr, ref_img)
  varnames = {};
  for j=1:length(name_arr)
    name = name_arr{j} + "_l2_err";
    varnames{end + 1} = name;
    err = comput_l2_err(ref_img, comp_models.(name_arr{j}).model.x);
    if (j == 1)
      tbl = table(err);
    else
      tbl = [tbl, table(err)];
    end
  end
  tbl.Properties.VariableNames = varnames;
end