clear
close
clc

dataName = 'aCGH_Pollack_chr17';

load(dataName);

[m,n] = size(D);
Omega0 = ~isnan(D);
D(isnan(D))=0;
opt = [];
[x] = aCGH_FLSA(D,Omega0);

plot(x(:,2),'b-');


