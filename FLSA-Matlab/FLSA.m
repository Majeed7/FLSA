function [x] = FLSA(y,lambda1,lambda2,tol)

if nargin < 5
    tol = 1e-4;
end

if lambda2 == 0
    x = sign(y).*max(y-alpha,0);
    return
end

if lambda1 == 0
    for j = 1:size(y,2)
        x(:,j) = flsa(y(:,j),zeros(size(y(:,j))),0,lambda2,size(y,1),100,tol,1,6);
    end
    return
end

for i=1:size(y,2)
    
    %mex file 
    x(:,i) = flsa(y(:,i),zeros(size(y(:,i))),lambda1,lambda2,size(y,1),100,1e-10,1,6);
    
    %% FOR Matlab ODE, decomment the lines below
    %init_point = zeros(n-1,1);
    %options = odeset('RelTol',tol,'AbsTol',tol);
    %[t,Z] = ode23tb(@FLSARNN_tv,[0 100],init_point,options,y(:,i),lambda1,lambda2,alpha);
    %z = Z(end,:)';
    %x(:,i) = y(:,i) - D'*z - projection(y(:,i) - D'*z,-lambda1,lambda1);
    %X = repmat(y(:,i),1,length(t)) - D'*Z' - projection(repmat(y(:,i),1,length(t)) - D'*Z',-lambda1,lambda1);
    
end


end


%The RNN Model
function dz = FLSARNN_tv(~,z,y,lambda1,lambda2,alpha)

ydz = y+[z;0]-[0;z];
x = max(-lambda1,min(ydz,lambda1)) - ydz;

dz = alpha*(max(-lambda2,min(lambda2,z+x(1:end-1)-x(2:end))) - z );

end


