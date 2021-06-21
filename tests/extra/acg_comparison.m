% Initialize
run('../../init.m');
seed = 777;
n = 2000;

% Loop over different choices of (mu, L) and generate a k-by-2 plot.
mu_vec = [1e-2, 1e-2, 1e-2, 1e-2];
L_vec =  [1e+1, 1e+2, 1e+3, 1e+4];
T_vec = sqrt(L_vec) * 1.5;

figure;
n_exp = min([length(mu_vec), length(L_vec)]);
n_rows = ceil(n_exp / 2);
for i=1:length(L_vec)
  mu = mu_vec(i);
  L = L_vec(i);
  T = T_vec(i);
  alg_hists = run_experiment(mu, L, n, T, seed);
  subplot(n_rows, 2, i);
  plot_hists(alg_hists)
  title(...
    ['$(T, \mu, L) = (' num2str(round(T, 1)) ',' ...
     num2str(mu) ',' num2str(L) ')$'], ...
     'interpreter', 'latex');
end

% Save the figure.
saveas(gcf, 'acg_cmp.fig');

%% Helper Functions

% Function to generate a nice looking plot.
function plot_hists(histories)

  % Plot
  max_obs = 200;
  lss = {'-', '-.'};
  fnames = fieldnames(histories);
  hold on;
  for i=1:length(fnames)
    iters = histories.(fnames{i}).iteration_values;
    log_vals = log(histories.(fnames{i}).function_values);
    nobs = length(iters);
    sub_idx = 1:ceil(nobs/max_obs):nobs;
    plot(iters(sub_idx), log_vals(sub_idx), lss{mod(i, 2)+1}, 'LineWidth', 1.1)
  end
  hold off;
  xlabel('Iteration Count')
  ylabel('$\log \phi(z)$', 'interpreter', 'latex')
  legend(fnames);
  
end


% Function to run a single experiment.
function histories = run_experiment(mu, L, n, time_limit, seed) 

  % Initialize,
  rng(seed);
  [U, ~] = qr(rand(n, n));
  S = diag(mu + (L - mu) * rand(n, 1));
  Q = U' * S * U;
  b = rand(n, 1);
  
  % Formulate the first-order oracle.
  quad_f_n = @(u) 0;
  quad_prox_f_n = @(u, lam) u;
  quad_f_s = @(u) (u - b)' * Q * (u - b) / 2;
  grad_quad_f_s = @(u) Q * (u - b);
  oracle = Oracle(quad_f_s, quad_f_n, grad_quad_f_s, quad_prox_f_n);
  
  % Set the base params.
  base = struct;
  base.L = L;
  base.mu = mu;
  base.x0 = zeros(n, 1);
  base.prod_fn = @(a, b) a' * b;
  base.norm_fn = @(a) sqrt(a' * a);
  base.iter_limit = Inf;
  base.time_limit = time_limit;
  base.i_logging = true;
  
  % Display experiment params.
  disp(table(mu, L, time_limit, n));
  
  % Run APG.
  disp('Running the ACG_s algorithm...')
  apg_params = base;
  [~, histories.ACG_s] = AdapAPG(oracle, apg_params);
  
  % Run ACG.
  disp('Running the ACG_f algorithm...')
  acg_params = base;
  [~, histories.ACG_f] = ACG(oracle, acg_params);
  
  % Post-processing
  disp('Done!')
end 
