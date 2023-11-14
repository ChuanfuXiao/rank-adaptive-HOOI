% Reconstruction of a low multilinear-rank tensor with Gaussian noise

I = 500; J = 500; K = 500;
P = 50; Q = 50; R = 50;

% generate the input tensor
U = rand(I,P); [U,~] = qr(U,0);
V = rand(J,Q); [V,~] = qr(V,0);
W = rand(K,R); [W,~] = qr(W,0);
G = rand(P,Q,R);
B = tmprod(G,{U,V,W},1:3); B = B/frob(B);
E = randn(I,J,K); E = E/frob(E);

SNR = 20;

delta = 1/10^(SNR/10);
A = B + delta*E;

epsilon = delta/frob(A);
