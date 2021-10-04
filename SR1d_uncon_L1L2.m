function [x,output] = SR1d_uncon_L1L2(N,fc,b,pm)
%min_x .5||S*F*x-b||^2 + lambda(|x|_1-|x|_2)

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
maxit = 5*N; maxoit = 10; 
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


for oit = 1:maxoit
    
     c = x/(norm(x,2)+eps); 
     xold = x;

   
    %ADMM method for solving the sub-problem
    for it = 1:maxit
            %update x
            rhs = ifft(Atb)*N  + lambda*c  + delta*(y+u);
            x = real(ifft(fft(rhs)./uker))/N; 
          
            %update y
            yold = y;
            y =shrink(x-u, lambda/delta);
  
            %update u
            u = u + y -x;
    
            % stopping condition for ADMM 
            r = norm(x-y);
            s = norm(delta*(y-yold));
            
            eps_pri = sqrt(N)*abstol + reltol*max(norm(x),norm(y));
            eps_dual = sqrt(N)*abstol + reltol*norm(u);
            
            if (r < eps_pri && s < eps_dual)
                break;
            end       
    end
        
     X = fft(x);
    
     Ax(1:fc+1) = X(1:fc+1);
     Ax(m-fc+1:m) = X(N-fc+1:N);
    
    
    %Stopping condition for DCA
    relerr = sqrt(sum((x-xold).^2))/max(sqrt(sum(x.^2)),1);
    
     output.L0(oit) = length(find(abs(x)>valuetol));
     output.obj(oit) = norm(x,1)-norm(x,2);
     output.res(oit) = norm(Ax-b)/norm(b);
     output.err(oit) = norm(x-xg)/norm(xg);
     output.relerr(oit) = norm(xold-x)/(norm(xold)+eps);
     
    if relerr < tol
        disp(['tolerance met after ' num2str(oit) ' iterations']);
        break;
    end
end


if (oit == maxoit)
    disp('Max outer iteration reached');
end



end



