% A class defintion for composite oracles
classdef Oracle < handle
  
  % -----------------------------------------------------------------------
  %% CONSTRUCTORS
  % -----------------------------------------------------------------------
  methods
    function obj = Oracle(varargin)
      
      if (nargin == 1) % eval constructor
        obj.o_eval_proxy_fn = varargin{1};        
      elseif (nargin == 4) % primary oracle constructor
        obj.o_eval_proxy_fn = @(x) struct(...
          'f_s_fn', varargin{1}, ...
          'grad_f_s_fn', varargin{2}, ...
          'f_n_fn', varargin{3}, ...
          'prox_f_n_fn', varargin{4});        
      else
        error('Incorrect number of arguments');
      end
      obj.eval_proxy_fn = obj.o_eval_proxy_fn;
      
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
  properties (Access = protected)
    f_s_at_prox_f_n
    f_n_at_prox_f_n
    grad_f_s_at_prox_f_n
  end
  
  % Evaluator oracles.
  properties (Access = protected, Hidden = true)
    o_eval_proxy_fn
    eval_proxy_fn
  end
  
  % -----------------------------------------------------------------------
  %% METHODS
  % -----------------------------------------------------------------------
  
  % Primary methods.
  methods
    
    % Evaluate the oracle at a point.
    function obj = eval(obj, x)
      oracle_outputs = obj.eval_proxy_fn(x);
      obj.f_s = @() oracle_outputs.f_s_fn(x);
      obj.grad_f_s = @() oracle_outputs.grad_f_s_fn(x);
      obj.f_n = @() oracle_outputs.f_n_fn(x);
      obj.prox_f_n = @(lam) oracle_outputs.prox_f_n_fn(x, lam);
      % Add hidden oracles if they exist.
      if isfield(oracle_outputs, 'grad_f_s_at_prox_f_n')
        obj.grad_f_s_at_prox_f_n = oracle_outputs.grad_f_s_at_prox_f_n;
      end
      if isfield(oracle_outputs, 'f_n_at_prox_f_n')
        obj.f_n_at_prox_f_n = oracle_outputs.f_n_at_prox_f_n;
      end
      if isfield(oracle_outputs, 'grad_f_s_at_prox_f_n')
        obj.grad_f_s_at_prox_f_n = oracle_outputs.grad_f_s_at_prox_f_n;
      end
    end
    
    % Scale the oracle by a constant
    function scale(obj, alpha)
      function scaled_struct = scaled_eval(x)
        o_struct = obj.o_eval_proxy_fn(x);
        scaled_struct = o_struct;
        scaled_struct.f_s = @() alpha * o_struct.f_s();
        scaled_struct.grad_f_s = @() alpha * o_struct.grad_f_s();
        scaled_struct.f_n = @() alpha * o_struct.f_n();
        scaled_struct.prox_f_n = @(lam) o_struct.prox_f_n(lam * alpha);
        % Account for hidden oracles.
        if isfield(o_struct, 'f_s_at_prox_f_n')
          scaled_struct.f_s_at_prox_f_n = @(lam) ...
            alpha * o_struct.f_s_at_prox_f_n(lam * alpha);
        end
        if isfield(o_struct, 'f_n_at_prox_f_n')
          scaled_struct.f_n_at_prox_f_n = @(lam) ...
            alpha * o_struct.f_n_at_prox_f_n(lam * alpha);
        end
        if isfield(o_struct, 'grad_f_s_at_prox_f_n')
          scaled_struct.grad_f_s_at_prox_f_n = @(lam) ...
            alpha * o_struct.grad_f_s_at_prox_f_n(lam * alpha);
        end
      end
      obj.eval_proxy_fn = @(x) scaled_eval(x);
    end
    
    % Add a prox term to the oracle at a point x_hat, i.e. add the function
    % (1/2) |x - x_hat| ^ 2. 
    function add_prox(obj, x_hat)
      function prox_struct = prox_eval(x)
        o_struct = obj.o_eval_proxy_fn(x);
        prox_struct = o_struct;
        prox_struct.f_s = @() ...
          o_struct.f_s() + (1 / 2) * obj.norm_fn(x - x_hat) ^ 2;
        prox_struct.grad_f_s = @() ...
          o_struct.grad_f_s() + (x - x_hat);
        % Account for hidden oracles.
        if isfield(o_struct, 'f_s_at_prox_f_n')
          prox_struct.f_s_at_prox_f_n = @(lam) ...
            o_struct.f_s_at_prox_f_n(lam) + ...
            (1 / 2) * obj.norm_fn(x - x_hat) ^ 2;
        end
        if isfield(o_struct, 'grad_f_s_at_prox_f_n')
          prox_struct.grad_f_s_at_prox_f_n = @(lam) ...
            o_struct.grad_f_s_at_prox_f_n(lam) + (x - x_hat);
        end
      end
      obj.eval_proxy_fn = @(x) prox_eval(x);
    end
    
    % Scale and add the prox term simultaneously.
    function proxify(obj, alpha, x_hat)
      obj.scale(alpha);
      obj.add_prox(x_hat);
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