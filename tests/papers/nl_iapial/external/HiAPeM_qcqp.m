function [x,z,out] = HiAPeM_qcqp(Q, c, d, m, x_l, x_u, opts) % nonconvex QCQP.

% ====================================================================
%
% Hybrid iALM and Penalty method for solving nonconvex QCQP
% min_x  .5*x'*Q{m+1}*x + c{m+1}'*x + d(m+1)
% s.t    .5*x'*Q{j}*x + c{j}'*x + d{j} <= 0, j = 1,...,m
%     lb(i) <= x(i) <= ub(i), i = 1,..., n
% Input:
%       model data: Q,c,d
%                    where Q = [Q(1),...,Q(m+1)],
%                          c = [c(1),...,c(m+1)],
%                          d = [d(1),...,d(m+1)].
%       lower and upper bound: x_l, x_u
%       algorithm parameters:
%           in_iter,
%           tol,
%           betamax
% Output:
%       x and y: primal-dual solution
%
% =====================================================================

nobj = 0; % numObj
ngrad = 0; % numGrad

% NEW: From William Kong.
if (~isfield(opts, 'termination_fn'))
    opts.termination_fn = [];
end
outer_iter = 1;

niter = 0;
n = length(x_l); %m = size(Q);

if isfield(opts,'K0')        K0 = opts.K0;             else  K0 = 100;           end
if isfield(opts,'K1')        K1 = opts.K1;             else  K1 = 2;             end
if isfield(opts,'sig')       sig = opts.sig;           else  sig = 3;            end
if isfield(opts,'gamma')     gamma = opts.gamma;       else  gamma = 1.1;        end
if isfield(opts,'beta0')     beta0 = opts.beta0;       else  beta0 = 1e-2;       end
if isfield(opts,'x0')        x0 = opts.x0;             else  x0 = randn(n,1);    end
if isfield(opts,'tol')       tol = opts.tol;           else  tol = 1e-3;         end
if isfield(opts,'maxit')     maxit = opts.maxit;       else  maxit = 100;        end
if isfield(opts,'APGmaxit')  APGmaxit = opts.APGmaxit; else  APGmaxit = 1000000; end % 10000
if isfield(opts,'adp')       adp = opts.adp;           else  adp = 1;            end
if isfield(opts,'frc')       frc = opts.frc;           else  frc = 0.5;          end
if isfield(opts,'inc')       inc = opts.inc;           else  inc = 1.5;          end
if isfield(opts,'dec')       dec = opts.dec;           else  dec = 2;            end
if isfield(opts,'rho')       rho = opts.rho;           else  rho = min(min(eig(Q{m+1})),0);        end

% NEW: By William Kong
if isfield(opts,'maxniter') maxniter = opts.maxniter;  else  maxniter = Inf;        end

x0 = min(max(x0, x_l),x_u);
x = x0;
z = zeros(m,1); % dual var
beta = beta0;

rho = max(abs(rho),tol); %

if isfield(opts,'Lip0')      Lip0_init = opts.Lip0;    else  Lip0_init = rho;    end

s = 1;

K_total = K0;

hist_numPG = [];

Qx = cell(1,m+1); Qxh = cell(1,m+1); 
gradf = cell(1,m+1); gradfh = cell(1,m+1);
fval = zeros(m+1,1); fvalh = zeros(m+1,1);

for j = 1:m+1
    Qx{j} = Q{j}*x;
    gradf{j} = Qx{j} + c{j};
    fval(j) = .5*x'*Qx{j} + c{j}'*x + d{j};
end

% start the initial stage by calling iALM
for k = 1:K0
    xk = x;
    
    [x,z,beta,total_PG_k] = iALM_str(x, xk, tol*frc, rho, beta0);
    
    hist_numPG = [hist_numPG; total_PG_k];
    
    for j = 1:m+1
        Qx{j} = Q{j}*x;
        gradf{j} = Qx{j} + c{j};
        fval(j) = .5*x'*Qx{j} + c{j}'*x + d{j};
    end

    gradL0 = Q{m+1}*x + c{m+1};
    for j = 1:m
        gradL0 = gradL0 + z(j)*gradf{j};
    end
    ngrad = ngrad + 1;
    I1 = (x == x_l); I2 = (x == x_u); I3 = (x > x_l & x < x_u);
    dres = norm( min(0,gradL0(I1)) )^2 + norm( max(0,gradL0(I2)) )^2 + norm(gradL0(I3))^2;
    dres = sqrt(dres);
    if (dres <= tol) % only check the dual_res because primal_res is below tol/2
      fprintf('succeed during the initial stage!\n');
      break;
    end
        
    % NEW: From William Kong
    if (niter >= maxniter)
      break
    end
    
    outer_iter = outer_iter + 1;
end

if dres > tol
    K_total = K0 + K1;
    s = s + 1;
    z_fix = z; 
    tol_pen = tol*frc; 
end

while dres > tol % only check dual_res since primal_res must below tol
    k = k + 1; beta_fix = beta;
    if k < K_total
        % call the penalty method
        xk = x;
        [x, z, beta,total_PG_k] = PenMM(xk, z_fix, rho, tol_pen, beta_fix);
        hist_numPG = [hist_numPG; total_PG_k];
        [~, feas] = feasibility(Q, c, d, x);
        
        % NEW: From William Kong
        if (isempty(opts.termination_fn))
          if (norm(x - xk) <= (1-frc)*tol/2/rho && feas <= tol)
            fprintf('final subproblem solved by PenMM\n');
            break;
          end
        else
          if (opts.termination_fn(x,z))
            break;
          end
        end
        outer_iter = outer_iter + 1;
    else
        % call the iALM method
        xk = x;
        [x,z,beta,total_PG_k] = iALM_str(x, xk, tol*frc, rho, beta0);
        hist_numPG = [hist_numPG; total_PG_k];
        [~, feas] = feasibility(Q, c, d, x);
        % NEW: From William Kong
        if (isempty(opts.termination_fn))
          if (norm(x - xk) <= (1-frc)*tol/2/rho && feas <= tol)
            fprintf('final subproblem solved by iALM\n');
            break;
          end
        else
          if (opts.termination_fn(x,z))
            break;
          end
        end
        outer_iter = outer_iter + 1;
        z_fix = z;
    end
    
    for j = 1:m+1
        Qx{j} = Q{j}*x;
        gradf{j} = Qx{j} + c{j};
        fval(j) = .5*x'*Qx{j} + c{j}'*x + d{j};
    end
    gradL0 = Q{m+1}*x + c{m+1};
    for j = 1:m
        gradL0 = gradL0 + z(j)*gradf{j};
    end
    ngrad = ngrad + 1;
    
    I1 = (x == x_l); I2 = (x == x_u); I3 = (x > x_l & x < x_u);
    
    dres = norm( min(0,gradL0(I1)) )^2 + norm( max(0,gradL0(I2)) )^2 + norm(gradL0(I3))^2;
    dres = sqrt(dres);
    
    if k == K_total & dres > tol
        Ks = ceil(gamma^s*K1);
        s = s + 1;
        K_total = K_total + Ks;
    end
    
    % NEW: From William Kong
    if (niter >= maxniter)
      break
    end
    
end

if k == K_total || s == 1
    fprintf('final subproblem solved by iALM\n');
else
    fprintf('final subproblem solved by PenMM\n');
end

if norm(x - xk) <= (1-frc)*tol/2/rho
    % compute dres in this case
    % otherwise dres already computed
    for j = 1:m+1
        Qx{j} = Q{j}*x;
        gradf{j} = Qx{j} + c{j};
        fval(j) = .5*x'*Qx{j} + c{j}'*x + d{j};
    end
    gradL0 = Q{m+1}*x + c{m+1};
    for j = 1:m
        gradL0 = gradL0 + z(j)*gradf{j};
    end
    ngrad = ngrad + 1;
    
    I1 = (x == x_l); I2 = (x == x_u); I3 = (x > x_l & x < x_u);
    
    dres = norm( min(0,gradL0(I1)) )^2 + norm( max(0,gradL0(I2)) )^2 + norm(gradL0(I3))^2;
    dres = sqrt(dres);
end

pres = norm( max(0, fval(1:m)) );

out.acg_ratio = niter / outer_iter;
out.nobj = nobj; % = 0 for non-adaptive!
out.ngrad = ngrad;
out.niter = niter;
out.obj = 1/2*x'*Q{m+1}*x+c{m+1}'*x;
out.dres = dres;
out.pres = pres;
out.numStage = s;
out.totalProb = k;
out.numPG = hist_numPG;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [x,z,beta,total_PG] = iALM_str(x0, xk, tol, rho, beta0)
        
        % =========================================
        % inexact ALM for solving k-subprob:
        %
        %       min f0(x) + rho|x-xk|^2, 
        %       s.t. .5*x'*Q{j}*x + c{j}'*x + d{j} <= 0, j = 1,...,m.
        %
        % where f0(x) = 0.5x'*Q(m+1)*x+c(m+1)'*x
        %
        % AL: f0(x)+rho|x-xk|^2 + 0.5/beta*(|[z+beta*f(x)]_+|^2-|z|^2)
        %
        % the output (x,z) will satisfy tol - KKT conditions
        %
        % ============================================
        x = x0;        z = zeros(m,1);    beta = beta0;
        
        tolk = 0.999 * min( tol, sqrt((sig-1)/sig*tol*rho)/2 );
        
        if adp
            [x, numPG] = APG_adp(x0, xk, z, rho, beta, tolk);  
        else
            [x, numPG] = APG(x0, xk, z, rho, beta, tolk);
        end
        
        for j = 1:m+1
          Qx{j} = Q{j}*x;
          gradf{j} = Qx{j} + c{j};
          fval(j) = .5*x'*Qx{j} + c{j}'*x + d{j};
        end

        Res = max(0, fval(1:m)); pres = norm(Res);
        z = max(0, z + beta*fval(1:m));
        total_PG = numPG;
        call_APG = 1;
        while pres > tol % only check primal_res since dual_res automatically below tol
            beta = beta*sig;
            if adp
                [x, numPG] = APG_adp(x, xk, z, rho, beta, tolk);  
            else
                [x, numPG] = APG(x, xk, z, rho, beta, tolk);
            end
            total_PG = total_PG + numPG;
            for j = 1:m+1
              Qx{j} = Q{j}*x;
              gradf{j} = Qx{j} + c{j};
              fval(j) = .5*x'*Qx{j} + c{j}'*x + d{j};
            end
            Res = max(0, fval(1:m)); pres = norm(Res);
            z = max(0, z + beta*fval(1:m));
            call_APG = call_APG + 1;
        end
        
        fprintf('%d-th subproblem by iALM succeed with %d APG calls and beta = %5.4f,\n',k, call_APG, beta);
    end % of iALM

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function [x,z,beta,total_PG] = PenMM(xk, zb, rho, tol, beta0)
        
        % =========================================
        % penalty method with estimated multiplier for solving k-subprob:
        %
        %       min f0(x) + rho|x-xk|^2, 
        %       s.t. .5*x'*Q{j}*x + c{j}'*x + d{j} <= 0, j = 1,...,m.
        %
        % where f0(x) = 0.5x'*Q(m+1)*x+c(m+1)'*x
        %
        % AL: f0(x)+rho|x-xk|^2 + 0.5/beta*(|[z+beta*f(x)]_+|^2-|z|^2)
        %
        % the output (x,z) will satisfy
        %           tol/2 - primal feasibility condition
        %           tol/2*min(1,sqrt(rho)) - dual feasibility condition
        %
        % ============================================
        
        x = xk; tolk = (1-frc)*tol/2*min(1, 1/sqrt(rho))*min(1,sqrt(rho));
        beta = beta0;
        
        if adp
            [x, numPG] = APG_adp(x, xk, zb, rho, beta, tolk);
        else
            [x, numPG] = APG(x, xk, zb, rho, beta, tolk);
        end
        for j = 1:m+1
              Qx{j} = Q{j}*x;
              gradf{j} = Qx{j} + c{j};
              fval(j) = .5*x'*Qx{j} + c{j}'*x + d{j};
        end
        Res = max(0, fval(1:m)); pres = norm(Res);
        
        total_PG = numPG;
        
        call_APG = 1;
        
        while pres > tol % only check primal_res since dual_res automatically below tolk
            beta = beta*sig;
            if adp
                [x, numPG] = APG_adp(x, xk, zb, rho, beta, tolk);
            else
                [x, numPG] = APG(x, xk, zb, rho, beta, tolk);
            end
            total_PG = total_PG + numPG;
            for j = 1:m+1
              Qx{j} = Q{j}*x;
              gradf{j} = Qx{j} + c{j};
              fval(j) = .5*x'*Qx{j} + c{j}'*x + d{j};
            end
            Res = max(0, fval(1:m)); pres = norm(Res);
            call_APG = call_APG + 1;
        end
        z = max(0, zb + beta*fval(1:m));
        
        fprintf('%d-th subproblem by PenMM succeed with %d APG calls and beta = %5.4f,\n',k, call_APG, beta);
        
    end % of PenMM

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [x, numPG] = APG_adp(x0, xk, z, rho, beta, tol) % new
        % scAPG from Q.Lin & L.Xiao, 2014
        
        % APG to solve
        %
        % min f0(x)+rho|x-xk|^2 + 0.5/beta*(|[z+beta*f(x)]_+|^2-|z|^2)
        %  where f0(x) = 0.5x'*Q(m+1)*x+c(m+1)'*x
        %
        % the output x will satisfy
        %       dist( 0, sub_diff(x) ) <= tol
        % use adaptive Lipschitz constant
        
        %inc = 2; % tune. 2.
        %dec = 2.5; % 2.5. >2 <3. Common choices: (2,2.5), (1.5,2), (2,2).
        
        % new AdapAPG: need Lip0 > rho!!
        %Lip0 = max(20*beta,rho); % tune. 20*beta good. >10*beta. also need > rho
        Lip0 = Lip0_init; % rho good. >= rho. Specify in test file, as global var.
        % initialization
        x = x0; xhat = x0;
        grad0 = Q{m+1}*x+c{m+1} +2*rho*(x-xk);
        for j = 1:m
            Qx{j} = Q{j}*x;
            gradf{j} = Qx{j} + c{j};
            fval(j) = .5*x'*Qx{j} + c{j}'*x + d{j};
            grad0 = grad0+ max(0,z(j)+beta*fval(j))*gradf{j};
        end
        niter = niter + 1;
        gradhat = grad0;
        I1 = (x == x_l); I2 = (x == x_u); I3 = (x > x_l & x < x_u);
        grad_res = norm( min(0,grad0(I1)) )^2 + norm( max(0,grad0(I2)) )^2 + norm(grad0(I3))^2;
        grad_res = sqrt(grad_res); hist_grad_res = grad_res;
        numPG = 0;
        alpha_old = 1; % new extrapolation weight.
        %obj = -norm(z)^2/2/beta; % Can ignore this part since indep. of x.
        while grad_res > tol && numPG < APGmaxit
            numPG = numPG + 1; Lip = Lip0/(dec*inc); % edited 12/4/2020
            objhat = 0; obj = inf; % arbitrary
            xtemp = x; % xtemp: x^k, x0: x^{k-1}
            while obj > objhat + gradhat'*(x - xhat) + 0.5*Lip*norm(x - xhat)^2 + 1e-10
                % now need compute 2 obj + 1 grad (instead of 1 obj) every search!
                Lip = Lip*inc;
                alpha = sqrt(rho/Lip); 
                extrap_weight = alpha*(1-alpha_old)/alpha_old/(1+alpha);
                xhat = xtemp + extrap_weight*(xtemp-x0);
                
                objhat = 0.5*xhat'*Q{m+1}*xhat+c{m+1}'*xhat +rho*norm(xhat-xk)^2;
                gradhat = Q{m+1}*xhat+c{m+1} +2*rho*(xhat-xk);
                for j = 1:m 
                    Qxh{j} = Q{j}*xhat; % at xhat
                    gradfh{j} = Qxh{j} + c{j}; % at xhat
                    fvalh(j) = .5*xhat'*Qxh{j} + c{j}'*xhat + d{j}; % at xhat
                
                    objhat = objhat + 0.5/beta*max(0, z(j)+beta*fvalh(j))^2;
                    gradhat = gradhat+ max(0,z(j)+beta*fvalh(j))*gradfh{j};
                end
                
                x = xhat - gradhat / Lip; x = min(x_u,max(x_l,x));
                
                obj = 0.5*x'*Q{m+1}*x+c{m+1}'*x +rho*norm(x-xk)^2;
                for j = 1:m 
                    Qx{j} = Q{j}*x; % at x
                    % gradf{j} = Qx{j} + c{j}; % at x. will compute out of loop.
                    fval(j) = .5*x'*Qx{j} + c{j}'*x + d{j}; % at x
                
                    obj = obj + 0.5/beta*max(0, z(j)+beta*fval(j))^2;
                end
                nobj = nobj + 2; ngrad = ngrad + 1; numPG = numPG + 1; 
                % NEW: From William Kong.
                niter = niter + 1;
            end
            x0 = xtemp; % xtemp: x^k, x0: x^{k-1}
            Lip0 = Lip; alpha_old = alpha;
            
            grad = Q{m+1}*x+c{m+1} +2*rho*(x-xk); %
            for j = 1:m
              % Qx{j} = Q{j}*x; % already computed
              gradf{j} = Qx{j} + c{j};
              % fval(j) = .5*x'*Qx{j} + c{j}'*x + d{j}; % already computed
              grad = grad+ max(0,z(j)+beta*fval(j))*gradf{j};
            end
            ngrad = ngrad + 1;
            I1 = (x == x_l); I2 = (x == x_u); I3 = (x > x_l & x < x_u);
            grad_res = norm( min(0,grad(I1)) )^2 + norm( max(0,grad(I2)) )^2 + norm(grad(I3))^2;
            grad_res = sqrt(grad_res);
            
            % NEW: From William Kong.
            niter = niter + 1;            
        end
        if numPG >= APGmaxit
            fprintf('APG_adp maxiter reached!\n');
        end
    end % end of APG_adp.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

% Utility function
function [residual_vec, feas] = feasibility(Q, c, d, x)
  m = length(Q) - 1;
  residual_vec = -Inf * ones(m, 1);
  for j=1:m
    residual_vec(j) = max([.5*x'*Q{j}*x + c{j}'*x + d{j}, 0]);
  end
  feas = norm(residual_vec);
end