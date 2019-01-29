function [B,info] = FLSAC(D,Omega,alpha,gamma,tol)


if nargin < 6
    tol = 1e-4;
end

if all(Omega(:))
    
    B = FLSA(D,alpha,gamma,tol);
    
    info.iter = 1;
    return;
elseif all(~Omega(:))
    B = zeros(size(D));
    info.iter = 1;
    return;
end

T = zeros(size(D,1)-1,size(D,1));
for i=1:size(D,1)-1
    T(i,i) = -1;
    T(i,i+1) = 1;
end


funVal = inf;
for iter = 1:50
    funValold = funVal;
    
    [B] = FLSA(D,alpha,gamma,tol);
    
    funVal = norm(D - B)^2 + alpha* norm(B,1) + gamma * norm(T*B,1);
    funValDiff = abs(funVal-funValold)/(funValold+eps);

    if funValDiff < tol
        break
    end
end

info.iter = iter;
info.funVal = funVal;
end