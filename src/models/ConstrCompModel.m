classdef ConstrCompModel < CompModel
  % An abstract model class for constrained composite optimization.
  %
  % Note:
  % 
  %   The following (non-inherited) properties are necessary before the 
  %   ``optimize()`` method can be called to solve the model: ``framework``,
  %   ``constr_fn``, ``grad_constr_fn``, ``set_projector``, and ``K_constr``.
  % 
  % Attributes:
  %
  %   framework (function handle): A **required** function handle to a
  %     framework that solves constrained composite optimization problems (see
  %     src.frameworks). Defaults to ``None``.
  %
  %   constr_fn (function handle): A **required** one argument function that,
  %     when evaluated at a point $x$, returns the constraint function at that 
  %     point, i.e., $g(x)$. Defaults to ``@(x) 0``.
  %
  %   grad_constr_fn (function handle): A **required** function that 
  %     represents the gradient of the constraint function. Has two possible
  %     prototypes: (i) a one argument function that, when evaluated at a
  %     point $x$, returns $\nabla g(x)$; and (ii) a two argument function
  %     that, when evaluated at $\{x, \delta\}$, returns $\nabla g(x) \delta$.
  %     Defaults to ``@(x) zeros(size(x))``.
  %
  %   set_projector (function handle): A one argument function
  %     that, when evaluated at a point $x$, returns the projection of $x$
  %     onto the set $S$. Defaults to ``@(x) zeros(size(x))``.
  %
  %   primal_cone_project (function handle): A one argument function that,
  %     that, when evaluated at a point $x$, returns the projection of $x$
  %     onto the cone K. Defaults to ``@(x) zeros(size(x))``.
  %
  %   dual_cone_projector (function handle): A one argument function that,
  %     that, when evaluated at a point $x$, returns the projection of $x$
  %     onto the dual cone K^{*}. Defaults to ``@(x) x``.
  %
  %   B_constr: A bound on the norm of the constraint function over the
  %     domain of $f_n$. Defaults to 0.0.
  %
  %   K_constr: A **required** Lipschitz constant of the constraint function.
  %     Defaults to ``None``.
  %
  %   L_constr: A Lipschitz constant of the gradient of the constraint
  %     function. Defaults to 0.0.
  %
  %   feas_tol (double): The tolerance for feasibility, i.e., 
  %     $\eta=\text{feas_tol}$. Defaults to ``1e-6``.
  %
  %   feas_type (character vector): Is either 'relative' or 'absolute'. If
  %     it is 'absolute', then the optimality condition is $\|w\|\leq 
  %     \text{opt_tol}$. If it is 'relative', then the optimality condition
  %     is $\|w\|/(1 + {\cal F}) \leq\text{opt_tol}$ where ${\cal F} = \| 
  %     g(x_0) - {\rm Proj}_S(g(x_0))\|$. Defaults to ``'absolute'``.
  %
  %   w (double vector): The feasibility residual returned by the solver.
  %     Defaults to ``None``. This property cannot be set by the user.

  % -----------------------------------------------------------------------
  %% GLOBAL PROPERTIES
  % -----------------------------------------------------------------------  
  
  % Visible global function properties.
  properties (SetAccess = public)
    framework
    constr_fn = @(x) 0;
    grad_constr_fn = @(x) zeros(size(x));
    set_projector = @(x) zeros(size(x));
    dual_cone_projector = @(x) x;
    feas_type {mustBeMember(feas_type, {'relative', 'absolute'})} = 'absolute'
  end
  
  % -----------------------------------------------------------------------
  %% SOLVER-RELATED PROPERTIES
  % -----------------------------------------------------------------------
  
  % Visible solver properties.
  properties (SetAccess = public)
    B_constr double {mustBeReal, mustBeFinite} = 0.0;
    K_constr double {mustBeReal, mustBeFinite}
    L_constr double {mustBeReal, mustBeFinite} = 0.0;
  end
  
  % Invisible and protected solver properties 
  properties (SetAccess = protected, Hidden = true)
    internal_feas_tol double {mustBeReal, mustBeFinite}
  end
  
  % -----------------------------------------------------------------------
  %% MODEL-RELATED PROPERTIES
  % -----------------------------------------------------------------------
  
  % Visible model properties.
  properties (SetAccess = protected)
    w double {mustBeReal}
  end
  
  % Invisible model properties.
  properties (SetAccess = protected, Hidden = true)
    norm_of_w double {mustBeReal}
  end
  
  % Invisible toleranace and limit properties.
  properties (SetAccess = public)
    feas_tol (1,1) double {mustBeReal, mustBePositive} = 1e-6
  end
    
  % -----------------------------------------------------------------------
  %% PRIMARY METHODS
  % -----------------------------------------------------------------------
  
  % Static methods.
  methods (Access = protected, Static = true, Hidden = true)
    % Converts status codes into strings
    function status_string = parse_status(status_num)
      status_string = parse_status@CompModel(status_num);
      if ~isempty(status_string)
        return;
      else
        status_map = containers.Map('KeyType', 'int32', 'ValueType', 'char');
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
  methods (Access = public, Hidden = true)
           
    % ---------------------------------------------------------------------
    %% MAIN OPTIMIZATION FUNCTIONS.
    % ---------------------------------------------------------------------
    
    % Compute the feasibility at a point
    function feas = compute_feas(obj, x)
      feas = obj.norm_fn(obj.constr_fn(x) - obj.set_projector(obj.constr_fn(x)));
    end
    
    % Key subroutines.
    function reset(obj)
      reset@CompModel(obj);
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
        if (obj.norm_of_v <= obj.internal_opt_tol && obj.norm_of_w <= obj.internal_feas_tol)
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
      obj.w = obj.model.w;
      obj.norm_of_w = obj.norm_fn(obj.w);
    end

    % Logging functions.
    function log_input(obj)
      log_input@CompModel(obj);
      word_len = 10;
      prefix = ['%-' num2str(word_len) 's = '];
      fprintf([prefix, '%s\n'], 'FRAMEWORK', func2str(obj.framework));
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
        obj.internal_feas_tol = obj.internal_feas_tol * (1 + obj.compute_feas(obj.x0));
      end
      obj.solver_input_params.feas_tol = obj.internal_feas_tol;
    end
    function update_solver_inputs(obj)
      update_solver_inputs@CompModel(obj);
      MAIN_INPUT_NAMES = ...
        {'B_constr', 'K_constr', 'L_constr', 'constr_fn', 'grad_constr_fn', 'set_projector', 'dual_cone_projector'};
      % Create the input params (order is important here!)
      for i=1:length(MAIN_INPUT_NAMES)
        NAME = MAIN_INPUT_NAMES{i};
        obj.solver_input_params.(NAME) = obj.(NAME);
      end
    end

  end

  % Public methods.
  methods (Access = public)
    
    % Main wrapper to optimize the model.
    function optimize(obj)
      obj.pre_process;
      if (obj.i_verbose)
        fprintf('\n');
        obj.log_input;
        fprintf('\n');
      end
      [obj.model, obj.history] = obj.framework(obj.solver, obj.oracle, obj.solver_input_params);
      obj.post_process;
      obj.get_status;
      if (obj.i_verbose)
        fprintf('\n');
        obj.log_output;
        fprintf('\n');
      end
    end

    % Viewing functions.
    function view_solution(obj)
      solns.x = obj.x;
      solns.v = obj.v;
      solns.w = obj.w;
      solns.f_at_x = obj.f_at_x;
      solns.norm_of_v = obj.norm_of_v;
      solns.norm_of_w = obj.norm_of_w;
      disp(solns);
    end
    
    function view_curvatures(obj)
      curvatures.L = obj.L;
      curvatures.M = obj.M;
      curvatures.m = obj.m;
      curvatures.K_constr = obj.K_constr;
      disp(curvatures);
    end

  end % End ordinary methods.

end % End of classdef