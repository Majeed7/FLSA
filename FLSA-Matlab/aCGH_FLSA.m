function [B,info] = aCGH_FLSA(D,Omega0,opt)

tic;
if nargin < 2
    Omega0 = true(size(D));
end    
if nargin < 3
    opt = struct();
end
if ~isfield(opt,'tol')
    opt.tol = 1e-4;
end

Omega0(isnan(D)) = false;
D(isnan(D)) = 0;

B = zeros(size(D));
if ~isfield(opt,'alpha') || ~isfield(opt,'gamma')
    sigma = 1.47*median(abs(D(Omega0(:))-median(D(Omega0(:)))));
    alpha_try = linspace(1,0.1,10)*sqrt(size(D,1))*sigma*0.2;
    gamma_try = linspace(1,0.1,10)*2*sigma;
    Omega1 = (rand(size(D))>1/3)&Omega0;
    Omega2 = ~Omega1&Omega0;
    % tune gamma
    minerr = inf;
    for j = 1:length(gamma_try)
        alpha = 0;
        gamma = gamma_try(j);
        B = FLSAC(D.*Omega1,Omega1,alpha,gamma,opt.tol);
        err = norm(Omega2.*(D-B),'fro');
        if err < minerr
            minerr = err;
            gamma_best = gamma;
        else
            break
        end
    end
    minerr = inf;
    for i = 1:length(alpha_try)
        alpha = alpha_try(i);
        gamma = gamma_best;
        B = FLSAC(D.*Omega1,Omega1,alpha,gamma,opt.tol);
        err = norm(Omega2.*(D-B),'fro');
        if err < minerr
            minerr = err;
            alpha_best = alpha;
        else
            break
        end

    end
else
    alpha_best = opt.alpha;
    gamma_best = opt.gamma;
end


B = FLSAC(D,Omega0,alpha_best,gamma_best,opt.tol);

info.alpha = alpha_best;
info.gamma = gamma_best;
info.time = toc;

fprintf("time:%f\n",info.time);
 end