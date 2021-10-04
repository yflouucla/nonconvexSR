function [x,output] = SR1d_uncon_L1(N,fc,b,pm)
%min_x .5||S*F*x-b||^2 + lambda|x|_1

%Input: signal dimension (N), frequency cut-off (fc), data (b), parameters
%set (pm)
%       pm.lambda: regularization paramter
%       pm.delta: penalty parameter for ADMM, default value: 10*lambda
%       pm.maxoit: max outer iterations, default value: 10
%       pm.maxit: max inner iterations: default value: 5000
%       pm.tol: outer tolerace, default value: 1e-3
%       pm.abstol: abs tolerance for ADMM: default value: 1e-7
%       pm.reltol: rel tolerance for ADMM: default value: 1e-5
%Output: reconstructed signal x
%       output.relerr: relative error of x_n and x_{n-1}
%       output.obj: objective function of x_n
%       output.res: residual norm(S*F*x_n-b)/norm(b)
%       output.err: error to the ground-truth norm(x_n-xg)/norm(xg)


%% parameters
lambda = 1e-5; detla = 10*lambda;
maxit = 2*N; maxoit = 10; 
tol = 1e-3; x0 = zeros(N,1); xg = x0;
abstol = 1e-7; reltol = 1e-5;
eps = 1e-16;
valuetol = 5e-4;

if isfield(pm,'delta'); delta = pm.delta; end
if isfield(pm,'lambda'); lambda = pm.lambda; end
if isfield(pm,'maxit'); maxit = pm.maxit; end
if isfield(pm,'maxoit'); maxoit = pm.maxoit; end
if isfield(pm,'x0'); x0 = pm.x0; end
if isfield(pm,'xg'); xg = pm.xg; end
if isfield(pm,'tol'); tol = pm.tol; end
if isfield(pm,'abstol'); abstol = pm.abstol; end
if isfield(pm,'reltol'); reltol = pm.reltol; end
if isfield(pm,'valuetol'); valuetol = pm.valuetol; end



%% pre-computing/initialize
m = 2*fc+1;
Mask = zeros(N,1); 
Mask(1:fc+1) = ones(fc+1,1);
Mask(N-fc+1:N) = ones(fc,1);

x = x0; 
u = zeros(N,1); y = u; Atb = u; 
uker = Mask + delta/N; 

Atb(1:fc+1) = b(1:fc+1);
Atb(N-fc+1:N) = b(m-fc+1:m);
Ax = b;


bz = b; Atx = zeros(N,1);
Atx(1:fc+1) = bz(1:fc+1);
Atx(N-fc+1:N) = bz(m-fc+1:m);

    
obj = @(x) .5*norm(Ax-b)^2+ lambda*(norm(x,1));


    for it = 1:maxit
        
  
        %update x
        xold = x;
        x =shrink(y-u, lambda/delta);

 
        %update y
        yold = y;
        rhs = ifft(Atx)*N + delta*(x+u);
        y = real(ifft(fft(rhs)./uker))/N; 
        
  
        %update u
        u = u + x-y;
        X = fft(x);

        Ax(1:fc+1) = X(1:fc+1);
        Ax(m-fc+1:m) = X(N-fc+1:N);
    

           
        relerr = norm(x-y)/max([norm(x),norm(y),eps]);
        output.relerr(it) = relerr;
        output.obj(it) = obj(x);
        output.res(it) = norm(Ax-b)/norm(b);
        output.err(it) = norm(x-xg)/norm(xg);
    
    
        if relerr < reltol  
            break;
        end
        
       
        
    end
end