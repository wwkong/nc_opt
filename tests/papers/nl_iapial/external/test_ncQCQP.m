clear; close all % Test code for nc_QCQP
%% generate data
rand('seed',20190927); randn('seed',20190927);
n = 1000; % 1000, 100
m = 10; % 10


num = 5;
rho_val = [0.1, 1, 10];
out1 = cell(length(rho_val),num);
out2 = cell(length(rho_val),num);
out3 = cell(length(rho_val),num);
out4 = cell(length(rho_val),num);

load result_ncQCQP_v1

for rho_num = 1:length(rho_val)
    rho = rho_val(rho_num);
    
    for ii = 1:num
        
        Q = cell(1,m+1);
        c = cell(1,m+1);
        d = zeros(m+1,1);
        
        [U,~] = qr(randn(n));
        
        v = max(0, randn(n,1)*5); % v ~ Gassian.
        Q{m+1} = U*diag(v)*U' - rho*eye(n);
        Q{m+1} = (Q{m+1} + Q{m+1}') / 2; % symmetrization is important
        c{m+1} = randn(n,1);
        
        fprintf('weak convexity constant is %f\n', min(eig(Q{m+1})));
        
        %%
        for j = 1:m
            [Q{j},~] = qr(randn(n));
            Q{j} = Q{j}(:, 1:n-5);
            v = rand(n-5,1)*5 + 1; 
            Q{j} = Q{j}*diag(v)*Q{j}';
            Q{j} = (Q{j} + Q{j}') / 2;
            c{j} = randn(n,1)*5;
            d(j) = min(0, randn*2) - 0.1;
        end
        
        lb = -5*ones(n,1); % can change.
        ub = 5*ones(n,1); % can change.
        
        maxsubit = 1000000;
        tol = 1e-3;
        
        %%
        opts = [];
        opts.x0 = zeros(n,1);
        opts.maxsubit = maxsubit;
        opts.maxit = 500;
        opts.sig = 3;
        opts.beta0 = 0.01;
        opts.tol = tol;
        opts.ver = 3; % opts.ver = 3; %% 1 for const beta
        opts.inc = 2;
        opts.dec= 2.5;
        opts.K0 = 100;
        
        t1 = tic;
        [x1,z,out1{rho_num,ii}] = HiAPeM_qcqp(Q,c,d,m,lb,ub,opts);
        time1 = toc(t1); out1{rho_num,ii}.time = time1;
        fprintf('HiAPeM time = %f\n', time1);
        
        opts.K0 = 10;
        
        t2 = tic;
        [x2,z,out2{rho_num,ii}] = HiAPeM_qcqp(Q,c,d,m,lb,ub,opts);
        time2 = toc(t2); out2{rho_num,ii}.time = time2;
        fprintf('HiAPeM time = %f\n', time2);
        
        opts.K0 = 1;
        
        t3 = tic;
        [x3,z,out3{rho_num,ii}] = HiAPeM_qcqp(Q,c,d,m,lb,ub,opts);
        time3 = toc(t3); out3{rho_num,ii}.time = time3;
        fprintf('HiAPeM time = %f\n', time3);
        
        %%
%         opts.beta0 = 10;
%         opts.K0 = 10000;
%         
%         t4 = tic;
%         [x4,out4{rho_num,ii}] = iPPP_qcqp(Q,c,d,m,lb,ub,opts);
%         time4 = toc(t4); out4{rho_num,ii}.time = time4;
%         fprintf('iPPP time = %f\n', time4);
    end
end

save result_ncQCQP out1 out2 out3 out4 m n