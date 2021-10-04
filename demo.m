clear; clc
close all;

%% problem setting
N = 1000;               % signal dimension
MS = 20;                % minimum separation
K = floor(N/(MS+1));    % sparsity
fc = 50;                % frequency cut-off


% generate sparse signal with MS
supp = randsample_separated(N,K,MS);
x = zeros(N,1);
xs = randn(K,1);
x(supp) = xs;
x_ref = x;


% data in frequecy domain with fc
m = 2*fc+1;
b = zeros(m,1);
Fx = fft(x);
b(1:fc+1) = Fx(1:fc+1);
b(m-fc+1:m) = Fx(N-fc+1:N);


%% algorithms
pm.lambda = 1e-6; 
pm.delta = 1e-5;
pm.alpha = 0.05;

x1 = SR1d_uncon_L1(N,fc,b,pm);
x12 = SR1d_uncon_L1L2(N,fc,b,pm);   
x1capped = SR1d_uncon_L1capped(N,fc,b,pm);

ErrL1 = norm(x1-x_ref)/norm(x_ref)
ErrL1L2 = norm(x12-x_ref)/norm(x_ref)
ErrL1capped = norm(x1capped-x_ref)/norm(x_ref)
        



