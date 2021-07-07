% Set up paths.
run('../../init.m');

%% Legend
alg_text = {'ECG', 'AIPP', 'AG', 'UP', 'NCF', 'IA', 'DA'};
times = [100, 200, 400, 800];

%% ANIME
disp('Table for the ANIME dataset');
load('anime_500K_506u_9437a_theta_history.mat')
out_tbl = generate_sgl_tbl(comp_history, times, 1);
load('anime_500K_506u_9437a_theta1e-01_history.mat')
out_tbl = [out_tbl; generate_sgl_tbl(comp_history, times, 0.1)];
load('anime_500K_506u_9437a_theta1e-02_history.mat')
out_tbl = [out_tbl; generate_sgl_tbl(comp_history, times, 0.01)];
disp(out_tbl)

%% FILM TRUST
disp('Table for the FILM TRUST dataset');
load('filmtrust_1508u_2071m_theta_history.mat')
out_tbl = generate_sgl_tbl(comp_history, times, 1);
load('filmtrust_1508u_2071m_theta1e-01_history.mat')
out_tbl = [out_tbl; generate_sgl_tbl(comp_history, times, 0.1)];
load('filmtrust_1508u_2071m_theta1e-02_history.mat')
out_tbl = [out_tbl; generate_sgl_tbl(comp_history, times, 0.01)];
disp(out_tbl)

%% TRUNCATED NORMAL
disp('Table for the TRUNCATED NORMAL dataset');
load('truncated_normal_theta_history.mat')
out_tbl = generate_sgl_tbl(comp_history, times, 1);
load('truncated_normal_theta1e-01_history.mat')
out_tbl = [out_tbl; generate_sgl_tbl(comp_history, times, 0.1)];
load('truncated_normal_theta1e-02_history.mat')
out_tbl = [out_tbl; generate_sgl_tbl(comp_history, times, 0.01)];
disp(out_tbl)

%% TRUNCATED NORMAL
disp('Table for the BINOMIAL dataset');
load('binomial_theta_history.mat')
out_tbl = generate_sgl_tbl(comp_history, times, 1);
load('binomial_theta1e-01_history.mat')
out_tbl = [out_tbl; generate_sgl_tbl(comp_history, times, 0.1)];
load('binomial_theta1e-02_history.mat')
out_tbl = [out_tbl; generate_sgl_tbl(comp_history, times, 0.01)];
disp(out_tbl)


%% Main table generating functions
function out_tbl = generate_sgl_tbl(data, times, theta)

  % Global settings  
  alg_name = fieldnames(data);
  tbl_names = {'theta', 't', alg_name{:}};
  out_tbl = ...
    array2table(zeros(length(times), length(tbl_names)));
  out_tbl.Properties.VariableNames = tbl_names;
  
  % vnorm values
  for it=1:length(times)
    out_tbl(it, 1) = table(theta);
    out_tbl(it, 2) = table(times(it));
    dpl = 2;
    for i=1:length(alg_name)
      a = alg_name{i};
      x_values = data.(a).time_values;
      y_values = cummin(data.(a).vnorm_values);
      idx = length(x_values(x_values <= times(it)));
      out_tbl(it, i + dpl) = table(y_values(idx));
    end
  end
  
end