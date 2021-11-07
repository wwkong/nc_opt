% Utility function that checks whether or not a pair (x,p) satisfies the first-order stationarity condition. 
%   - grad_constr_fn(a,b) is the gradient (or derivative) of the constraint function at `a` evaluated with displacement `b`.
%   - proj_dh(a,b) is the projection of `b` onto the subdifferential of h at `a`.
%   - proj_NKt(a,b) is the projection of `b` onto the normal cone of the dual cone of K at `a`.
function [terminate, w, q] = termination_check(x, p, oracle, constr_fn, grad_constr_fn, proj_dh, proj_NKt, norm_fn, rho, eta) 
  
  % Initialize.
  o_at_x = oracle.eval(x);
  primal_subgrad = -o_at_x.grad_f_s() - grad_constr_fn(x, p);
  w = proj_dh(x, primal_subgrad) - primal_subgrad;
  primal_residual = norm_fn(w);
  dual_subgrad = constr_fn(x);
  q = proj_NKt(p, dual_subgrad) - dual_subgrad;
  dual_residual = norm_fn(q);
  
  % Check and output.
  if (primal_residual <= rho && dual_residual <= eta)
    terminate = true;
  else
    terminate = false;
  end
end