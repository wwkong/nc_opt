% Set up paths.
run('../../init.m');

%% Legend
leg_text = {'ECG', 'AIPP', 'AG', 'UP', 'NCF', 'IA', 'DA'};

%% Real-World Data
load('anime_500K_506u_9437a_theta1e-01_history.mat')
generate_sgl_plot(comp_history, 'anime_500k', [0, 1000], leg_text)

%% Synthetic Data
load('truncated_normal_theta1e-01_history.mat')
generate_sgl_plot(comp_history, 'truncated_normal', [0, 1000], leg_text)

%% Main plotting functions
function generate_sgl_plot(data, fname, xlim_vec, leg_text)

  % Global settings  
  alg_name = fieldnames(data);
  lns = lines(9);
  lns(3, :) = [];
  line_width = 1.1;
  line_specs = {'-s', '-^', '-+', ':', '-', '-.', '--'};
  color_specs = lns;
  marker_size = 4;
  max_obs = 35;
  
  % Figure
  figure;
  set(gcf, ...
     'DefaultAxesLineStyleOrder', line_specs, ...
     'DefaultAxesColorOrder', color_specs, ...
     'DefaultLineLineWidth', line_width);
  
  % Function values
  set(subplot(1, 2, 1), 'OuterPosition', [0.0, 0.10, 0.5, 0.90]);
  ax = gca;
  hold on;
  for i=1:length(alg_name)
    a = alg_name{i};
    x_values = data.(a).time_values;
    y_values = cummin(data.(a).function_values);
    dspace = 1 + floor(length(x_values) / max_obs);
    plot(...
      x_values(1:dspace:end), ...
      log(y_values(1:dspace:end)), ...
      'MarkerSize', marker_size);
    title('Function Value');
    xlabel('Time (in seconds)', 'interpreter', 'latex')
    ylabel(...
      '$$\log \left[ \phi(z_k) \right]$$', ...
      'interpreter', 'latex');
    xlim(xlim_vec);
    ax.LineStyleOrderIndex = ax.ColorOrderIndex;
  end
  hold off;
  
  % vnorm values
  set(subplot(1, 2, 2), 'OuterPosition', [0.5, 0.10, 0.5, 0.90]);
  hold on;
  ax = gca;
  for i=1:length(alg_name)
    a = alg_name{i};
    x_values = data.(a).time_values;
    y_values = cummin(data.(a).vnorm_values);
    dspace = 1 + floor(length(x_values) / max_obs);
    plot(...
      x_values(1:dspace:end), ...
      log(y_values(1:dspace:end)), ...
      'MarkerSize', marker_size);
    title('Subgradient Size');
    xlabel('Time (in seconds)', 'interpreter', 'latex')    
    ylabel(...
      '$$\log \left[\min_{i\leq k} \|v_i\| \right]$$', ...
      'interpreter', 'latex');
    xlim(xlim_vec);
    ax.LineStyleOrderIndex = ax.ColorOrderIndex;
  end
  hold off;

  % Legend
  leg = legend(leg_text, 'interpreter', 'none', ...
      'Location','SouthOutside','Orientation','horizontal');
  set(leg, 'position', [0.125, 0.025, 0.75, 0.05]);
  
  % Output
  width = 8;
  height = 3;
  fig_posn = [0 0 width height];
  fig = gcf;
  fig.PaperPosition = fig_posn;
  print(fname, '-dsvg')
  
end