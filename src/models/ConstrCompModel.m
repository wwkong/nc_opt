% A class defintion for constrained composite optimization models
classdef ConstrCompModel < CompModel
      
  % -----------------------------------------------------------------------
  %% GLOBAL PROPERTIES
  % -----------------------------------------------------------------------  
  
  % Visible global function properties.
  properties (SetAccess = public)
    framework
    constr_fn = @(x) 0;
    grad_constr_fn = @(x) zeros(size(x));
    set_projector = @(x) 0;
    feas_type {mustBeMember(feas_type, {'relative', 'absolute'})} = ...
      'absolute'
  end
  
  % -----------------------------------------------------------------------
  %% SOLVER-RELATED PROPERTIES
  % -----------------------------------------------------------------------
  
  % Visible solver properties.
  properties (SetAccess = public)
    K_constr double {mustBeReal, mustBeFinite}
  end
  
  % Invisible and protected solver properties 
  properties (SetAccess = protected, Hidden = true)
    internal_feas_tol
  end
  
  % -----------------------------------------------------------------------
  %% MODEL-RELATED PROPERTIES
  % -----------------------------------------------------------------------
  
  % Visible model properties.
  properties (SetAccess = protected)
    y double {mustBeReal, mustBeFinite}
    w double {mustBeReal, mustBeFinite}
  end
  
  % Invisible model properties.
  properties (SetAccess = protected, Hidden = true)
    norm_of_w double {mustBeReal, mustBeFinite}
  end
  
  % Invisible toleranace and limit properties.
  properties (SetAccess = public, Hidden = true)
    feas_tol (1,1) double {mustBeReal, mustBePositive} = 1e-6
  end
    
  % -----------------------------------------------------------------------
  %% PRIMARY METHODS
  % -----------------------------------------------------------------------
  
  % Static methods.
  methods (Access = protected, Static = true)
    % Converts status codes into strings
    function status_string = parse_status(status_num)
      status_string = parse_status@CompModel(status_num);
      if ~isempty(status_string)
        return;
      else
        status_map = ...
          containers.Map('KeyType', 'int32', 'ValueType', 'char');
        status_map(-202) = 'ALL_STOPPING_CONDITONS_FAILED';
        status_map(-201) = 'FEASIBILITY_CONDITION_FAILED';
        status_map(201)  = 'ALL_STOPPING_CONDITONS_ACHIEVED';
        if isKey(status_map, status_num)
          status_string = status_map(status_num);
        else
          status_string = [];
        end
      end
    end  
  end
  
  
  % Ordinary methods.
  methods (Access = public)
           
    % ---------------------------------------------------------------------
    %% MAIN OPTIMIZATION FUNCTIONS.
    % ---------------------------------------------------------------------
    
    % Compute the feasibility at a point
    function feas = compute_feas(obj, x)
      feas = obj.norm_fn(...
        obj.constr_fn(x) - ...
        obj.set_projector(obj.constr_fn(x)));
    end
    
    % Key subroutines.
    function reset(obj)
      reset@CompModel(obj);
      obj.y = [];
      obj.w = [];
    end
    function check_inputs(obj)
      check_inputs@CompModel(obj);
      obj.check_field('K_constr');
      if (obj.K_constr == 0.0)
        error('The Lipschitz constant K_constr cannot be zero!');
      end
    end
    function get_status(obj)
      get_status@CompModel(obj)
      if (~ismember(obj.status, [103, 104]))
        if (obj.norm_of_v <= obj.internal_opt_tol && ...
            obj.norm_of_w <= obj.internal_feas_tol)
          obj.status = 201; % ALL_STOPPING_CONDITONS_ACHIEVED
        elseif (obj.norm_of_v > obj.internal_opt_tol)
          obj.status = -101; % STATIONARITY_CONDITION_FAILED
        elseif (obj.norm_of_w > obj.internal_feas_tol)
          obj.status = -201; % FEASIBILITY_CONDITION_FAILED
        else
          obj.status = -202; % ALL_STOPPING_CONDITONS_FAILED
        end
      end
    end
    function post_process(obj)
      post_process@CompModel(obj)
      obj.y = obj.model.y;
      obj.w = obj.model.w;
      obj.norm_of_w = obj.norm_fn(obj.w);
    end
    
    % Main wrapper to optimize the model.
    function optimize(obj)
      obj.pre_process;
      if (obj.i_verbose)
        fprintf('\n');
        obj.log_input;
        fprintf('\n');
      end
      [obj.model, obj.history] = ...
        obj.framework(obj.solver, obj.oracle, obj.solver_input_params);
      obj.post_process;
      obj.get_status;
      if (obj.i_verbose)
        fprintf('\n');
        obj.log_output;
        fprintf('\n');
      end
    end

    % Logging functions.
    function log_input(obj)
      log_input@CompModel(obj);
      word_len = 8;
      prefix = ['%-' num2str(word_len) 's = '];
      fprintf([prefix, '%.2e\n'], 'FEAS_TOL', obj.feas_tol);
    end
    function log_output(obj)
      log_output@CompModel(obj);
      word_len = 16;
      prefix = ['%-' num2str(word_len) 's = '];
      fprintf([prefix, '%.2e\n'], 'NORM OF W', obj.norm_of_w);
    end
    
    % Sub-update functions.
    function update_tolerances(obj)
      update_tolerances@CompModel(obj);
      obj.internal_feas_tol = obj.feas_tol;
      if strcmp(obj.feas_type, 'relative')
        obj.internal_feas_tol = ...
          obj.internal_feas_tol * (1 + obj.compute_feas(obj.x0));
      end
      obj.solver_input_params.feas_tol = obj.internal_feas_tol;
    end
    function update_solver_inputs(obj)
      update_solver_inputs@CompModel(obj);
      MAIN_INPUT_NAMES = {...
        'K_constr', 'constr_fn', 'grad_constr_fn', 'set_projector'
      };
      % Create the input params (order is important here!)
      for i=1:length(MAIN_INPUT_NAMES)
        NAME = MAIN_INPUT_NAMES{i};
        obj.solver_input_params.(NAME) = obj.(NAME);
      end
    end
    
  end % End ordinary methods.

end % End of classdef