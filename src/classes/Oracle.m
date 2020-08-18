% A class defintion for composite oracles; inherits from a copyable 
% handle superclass.
classdef Oracle < matlab.mixin.Copyable
  
  % -----------------------------------------------------------------------
  %% CONSTRUCTORS
  % -----------------------------------------------------------------------
  methods
    function obj = Oracle(varargin)
      if (nargin == 1) % eval constructor
        obj.eval_proxy_fn = varargin{1};        
      elseif (nargin == 4) % basic oracle constructor
        obj.eval_proxy_fn = @(x) struct(...
          'f_s', @() feval(varargin{1}, x), ...
          'f_n', @() feval(varargin{2}, x), ...
          'grad_f_s', @() feval(varargin{3}, x), ...
          'prox_f_n', @(lam) feval(varargin{4}, x, lam));        
      else
        error('Incorrect number of arguments');
      end
    end
  end
  
  % -----------------------------------------------------------------------
  %% ORACLES
  % -----------------------------------------------------------------------
  
  % Primary oracles (user-side).
  properties (SetAccess = public)
    f_s
    grad_f_s
    f_n
    prox_f_n
  end
  
  % Invisible global function properties.
  properties (Access = public, Hidden = true)
    prod_fn = @(a,b) sum(dot(a, b))
    norm_fn = @(a) norm(a, 'fro')
  end
  
  % Hidden secondary oracles.
  properties (SetAccess = protected)
    f_s_at_prox_f_n
    f_n_at_prox_f_n
    grad_f_s_at_prox_f_n
  end
  
  % Evaluator oracles.
  properties (SetAccess = protected, Hidden = true)
    eval_proxy_fn
  end
  
  % -----------------------------------------------------------------------
  %% METHODS
  % -----------------------------------------------------------------------
  
  % Helper methods that depend on the object.
  methods
    
    % Adds a prox term of the form (1 / 2) * ||x - x_hat|| ^2 
    % to an oracle struct.
    function prox_struct = add_prox_to_struct(obj, base_struct, x, x_hat)
        prox_struct = base_struct;
        prox_struct.f_s = @() ...
          base_struct.f_s() + (1 / 2) * obj.norm_fn(x - x_hat) ^ 2;
        prox_struct.grad_f_s = @() ...
          base_struct.grad_f_s() + (x - x_hat);
        % Account for hidden oracles.
        if isfield(base_struct, 'f_s_at_prox_f_n')
          prox_struct.f_s_at_prox_f_n = @(lam) ...
            base_struct.f_s_at_prox_f_n(lam) + ...
            (1 / 2) * obj.norm_fn(x - x_hat) ^ 2;
        end
        if isfield(base_struct, 'grad_f_s_at_prox_f_n')
          prox_struct.grad_f_s_at_prox_f_n = @(lam) ...
            base_struct.grad_f_s_at_prox_f_n(lam) + (x - x_hat);
        end
    end  
  end
  
  % Helper methods that are independent of the object.
  methods (Static = true)
      
    % Scales an oracle by alpha.
    function scaled_struct = scale_struct(base_struct, alpha)
      scaled_struct = base_struct;
      scaled_struct.f_s = @() alpha * base_struct.f_s();
      scaled_struct.grad_f_s = @() alpha * base_struct.grad_f_s();
      scaled_struct.f_n = @() alpha * base_struct.f_n();
      scaled_struct.prox_f_n = @(lam) base_struct.prox_f_n(lam * alpha);
      % Account for hidden oracles.
      if isfield(base_struct, 'f_s_at_prox_f_n')
        scaled_struct.f_s_at_prox_f_n = @(lam) ...
          alpha * base_struct.f_s_at_prox_f_n(lam * alpha);
      end
      if isfield(base_struct, 'f_n_at_prox_f_n')
        scaled_struct.f_n_at_prox_f_n = @(lam) ...
          alpha * base_struct.f_n_at_prox_f_n(lam * alpha);
      end
      if isfield(base_struct, 'grad_f_s_at_prox_f_n')
        scaled_struct.grad_f_s_at_prox_f_n = @(lam) ...
          alpha * base_struct.grad_f_s_at_prox_f_n(lam * alpha);
      end
    end
    
  end
  
  % Primary methods.
  methods
    
    % Evaluate the oracle at a point.
    function obj = eval(obj, x)
      oracle_outputs = obj.eval_proxy_fn(x);
      obj.f_s = @() oracle_outputs.f_s();
      obj.f_n = @() oracle_outputs.f_n();
      obj.grad_f_s = @() oracle_outputs.grad_f_s();
      obj.prox_f_n = @(lam) oracle_outputs.prox_f_n(lam);
      % Add hidden oracles if they exist.
      if isfield(oracle_outputs, 'f_s_at_prox_f_n')
        obj.f_s_at_prox_f_n = ...
          @(lam) oracle_outputs.f_s_at_prox_f_n(lam);
      end
      if isfield(oracle_outputs, 'f_n_at_prox_f_n')
        obj.f_n_at_prox_f_n = ...
          @(lam) oracle_outputs.f_n_at_prox_f_n(lam);
      end
      if isfield(oracle_outputs, 'grad_f_s_at_prox_f_n')
        obj.grad_f_s_at_prox_f_n = ...
          @(lam) oracle_outputs.grad_f_s_at_prox_f_n(lam);
      end
    end
    
    % Scale the oracle by a constant
    function scale(obj, alpha)
      eval_proxy_fn_cpy = obj.eval_proxy_fn;
      function out_struct = scaled_eval(x)
        out_struct = eval_proxy_fn_cpy(x);
        out_struct = obj.scale_struct(out_struct, alpha);
      end
      obj.eval_proxy_fn = @(x) scaled_eval(x);
    end
      
    % Add a prox term to the oracle at a point x_hat, i.e. add the function
    % (1/2) ||x - x_hat|| ^ 2. 
    function add_prox(obj, x_hat)
      eval_proxy_fn_cpy = obj.eval_proxy_fn;
      function out_struct = prox_eval(x)
        out_struct = eval_proxy_fn_cpy;
        out_struct = obj.add_prox_to_struct(out_struct, x, x_hat);
      end
      obj.eval_proxy_fn = @(x) prox_eval(x);
    end
    
    % Scale and add the prox term simultaneously.
    function proxify(obj, alpha, x_hat)
      eval_proxy_fn_cpy = obj.eval_proxy_fn;
      function combined_struct = proxify_eval(x)
        out_struct = eval_proxy_fn_cpy(x);
        scaled_struct = obj.scale_struct(out_struct, alpha);
        combined_struct = obj.add_prox_to_struct(scaled_struct, x, x_hat);
      end
      obj.eval_proxy_fn = @(x) proxify_eval(x);
    end
    
    % Add the smooth part of an input oracle to this oracle
    function add_smooth_oracle(obj, input_oracle)
      eval_proxy_fn_cpy = obj.eval_proxy_fn;
      function combined_struct = combined_eval(x)
        o_struct = eval_proxy_fn_cpy(x);
        input_struct = input_oracle.eval(x);
        combined_struct = o_struct;
        combined_struct.f_s = @() o_struct.f_s() + input_struct.f_s();
        combined_struct.grad_f_s = @() ...
          o_struct.grad_f_s() + input_struct.grad_f_s();
        % Account for hidden oracles.
        if isfield(o_struct, 'f_s_at_prox_f_n')
          combined_struct.f_s_at_prox_f_n = @(lam) ...
            o_struct.f_s_at_prox_f_n(lam) + ...
            input_struct.f_s_at_prox_f_n(lam);
        end
        if isfield(o_struct, 'grad_f_s_at_prox_f_n')
          combined_struct.grad_f_s_at_prox_f_n = @(lam) ...
            o_struct.grad_f_s_at_prox_f_n(lam) + ...
            input_struct.grad_f_s_at_prox_f_n(lam);
        end
      end
      obj.eval_proxy_fn = @(x) combined_eval(x);
    end
    
    % Obtain the individual oracle components
    function [f_s, f_n, grad_f_s, prox_f_n] = decompose(obj)

      % Single-input functions
      f_s = @(x) feval(subsref(obj.eval(x), ...
        struct('type','.','subs','f_s')));
      f_n = @(x) feval(subsref(obj.eval(x), ...
        struct('type','.','subs','f_n')));
      grad_f_s = @(x) feval(subsref(obj.eval(x), ...
        struct('type','.','subs','grad_f_s')));

      % Multi-input functions
      prox_f_n = @(x, lam) feval(subsref(obj.eval(x), ...
        struct('type','.','subs','prox_f_n')), lam);

    end
    
  end
  
end