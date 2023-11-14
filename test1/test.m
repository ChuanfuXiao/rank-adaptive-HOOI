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

%%

% the error tolerance
% epsilon = 1.0e-6;

% PSNR = 35;
% I = 512; J = 512;
% % A = A(:,:,:,50);
% normA = frob(A);
% epsilon = (255*sqrt(3*I*J))/(10^(PSNR/20)*normA);

% % initialization
sizeA = size(A);
N = length(sizeA);
% U_init = cell(1,N);
% 
% for n = 1:N
%     Q = randn(sizeA(n),sizeA(n));
%     [Q,~] = qr(Q,0);
%     U_init{n} = Q;
% end

% t-hosvd
tStart = cputime; [U_t,G_t] = mlsvd(A,epsilon,0); t_t = cputime - tStart;

B_hat = tmprod(G_t,U_t,1:N);
res_t = frob(B_hat - B)/frob(B);

% G_t = tmprod(G_t,U_t{3},3);
% ratio_t = prod(size(A))/prod(size(G_t));
% tStart = cputime; ca_t = classification(U_t{1},G_t,X,labels); ta_t = cputime - tStart;

% para_t = prod(size(G_t))+prod(size(U_t{1}))+prod(size(U_t{2}))+prod(size(U_t{3}));

% para_t = prod(size(G_t))+prod(size(U_t{1}))+prod(size(U_t{2}))+prod(size(U_t{3}))+prod(size(U_t{4}));

% % random t-hosvd
% tStart = cputime; [T_rt1,ranks_rt1] = rand_hosvd(A,epsilon,'sequential',false,'rand','randQB'); t_rt1 = cputime - tStart;
% % B_hat = tmprod(T_rt1.core,T_rt1.U,1:N);
% % res_rt1 = frob(B_hat - B)/frob(B);
% ratio_rt1 = prod(size(A))/prod(size(T_rt1.core));
% % para_rt1 = prod(size(T_rt1.core))+prod(size(T_rt1.U{1}))+prod(size(T_rt1.U{2}))+prod(size(T_rt1.U{3}))+prod(size(T_rt1.U{4}));
% 
% tStart = cputime; [T_rt2,ranks_rt2] = rand_hosvd(A,epsilon,'sequential',false,'rand','randBGKL'); t_rt2 = cputime - tStart;
% % B_hat = tmprod(T_rt2.core,T_rt2.U,1:N);
% % res_rt2 = frob(B_hat - B)/frob(B);
% ratio_rt2 = prod(size(A))/prod(size (T_rt2.core));
% % para_rt2 = prod(size(T_rt2.core))+prod(size(T_rt2.U{1}))+prod(size(T_rt2.U{2}))+prod(size(T_rt2.U{3}))+prod(size(T_rt2.U{4}));

% st-hosvd
tStart = cputime; [U_st,G_st] = mlsvd(A,epsilon); t_st = cputime - tStart;

B_hat = tmprod(G_st,U_st,1:N);
res_st = frob(B_hat - B)/frob(B);

% G_st = tmprod(G_st,U_st{3},3);
% ratio_st = prod(size(A))/prod(size(G_st));
% tStart = cputime; ca_st = classification(U_st{1},G_st,X,labels); ta_st = cputime - tStart;
% para_st = prod(size(G_st))+prod(size(U_st{1}))+prod(size(U_st{2}))+prod(size(U_st{3}));

% para_st = prod(size(G_st))+prod(size(U_st{1}))+prod(size(U_st{2}))+prod(size(U_st{3}))+prod(size(U_st{4}));

% % random st-hosvd
% tStart = cputime; [T_rs1,ranks_rs1] = rand_hosvd(A,epsilon,'sequential',true,'rand','randQB'); t_rs1 = cputime - tStart;
% % B_hat = tmprod(T_rs1.core,T_rs1.U,1:N);
% % res_rs1 = frob(B_hat - B)/frob(B);
% ratio_rs1 = prod(size(A))/prod(size(T_rs1.core));
% % para_rs1 = prod(size(T_rs1.core))+prod(size(T_rs1.U{1}))+prod(size(T_rs1.U{2}))+prod(size(T_rs1.U{3}))+prod(size(T_rs1.U{4}));
% 
% tStart = cputime; [T_rs2,ranks_rs2] = rand_hosvd(A,epsilon,'sequential',true,'rand','randBGKL'); t_rs2 = cputime - tStart;
% % B_hat = tmprod(T_rs2.core,T_rs2.U,1:N);
% % res_rs2 = frob(B_hat - B)/frob(B);
% ratio_rs2 = prod(size(A))/prod(size(T_rs2.core));
% % para_rs2 = prod(size(T_rs2.core))+prod(size(T_rs2.U{1}))+prod(size(T_rs2.U{2}))+prod(size(T_rs2.U{3}))+prod(size(T_rs2.U{4}));

% greedy-hosvd
tStart = cputime; [T_g,ranks_g] = greedy_hosvd(A,epsilon); t_g = cputime - tStart;

B_hat = tmprod(T_g.core,T_g.U,1:N);
res_g = frob(B_hat - B)/frob(B);

% G_g = tmprod(T_g.core,T_g.U{3},3);
% ratio_g = prod(size(A))/prod(size(G_g));
% tStart = cputime; ca_g = classification(T_g.U{1},G_g,X,labels); ta_g = cputime - tStart;
% para_g = prod(size(T_g.core))+prod(size(T_g.U{1}))+prod(size(T_g.U{2}))+prod(size(T_g.U{3}));

% para_g = prod(size(T_g.core))+prod(size(T_g.U{1}))+prod(size(T_g.U{2}))+prod(size(T_g.U{3}))+prod(size(T_g.U{4}));

% rank-adaptive hooi
tStart = cputime; [T_rah,ranks_rah,iter] = rank_ada_hooi(A,epsilon); t_rah = cputime - tStart;

B_hat = tmprod(T_rah.core,T_rah.U,1:N);
res_rah = frob(B_hat - B)/frob(B);

% G_rah = tmprod(T_rah.core,T_rah.U{3},3);
% ratio_rah = prod(size(A))/prod(size(G_rah));
% tStart = cputime; ca_rah = classification(T_rah.U{1},G_rah,X,labels); ta_rah = cputime - tStart;
% para_rah = prod(size(T_rah.core))+prod(size(T_rah.U{1}))+prod(size(T_rah.U{2}))+prod(size(T_rah.U{3}));

% para_rah = prod(size(T_rah.core))+prod(size(T_rah.U{1}))+prod(size(T_rah.U{2}))+prod(size(T_rah.U{3}))+prod(size(T_rah.U{4}));

% improve rank-adaptive hooi
% tStart = cputime; [T_irah,ranks_irah,iiter] = irank_ada_hooi(A,epsilon); t_irah = cputime - tStart;
% 
% % B_hat = tmprod(T_irah.core,T_irah.U,1:N);
% % res_irah = frob(B_hat - B)/frob(B);
% 
% G_irah = tmprod(T_irah.core,T_irah.U{3},3);
% ratio_irah = prod(size(A))/prod(size(G_irah));
% tStart = cputime; ca_irah = classification(T_irah.U{1},G_irah,X,labels); ta_irah = cputime - tStart;

% para_irah = prod(size(T_irah.core))+prod(size(T_irah.U{1}))+prod(size(T_irah.U{2}))+prod(size(T_irah.U{3}))+prod(size(T_irah.U{4}));

% % tStart = cputime; [T_iirah,ranks_iirah,iiiter] = iirank_ada_hooi(A,epsilon); t_iirah = cputime - tStart;
% % 
% % B_hat = tmprod(T_irah.core,T_irah.U,1:N);
% % res_irah = frob(B_hat - B)/frob(B);
% % 
% % G_iirah = tmprod(T_iirah.core,T_iirah.U{3},3);
% % ratio_iirah = prod(size(A))/prod(size(G_iirah));
% % tStart = cputime; ca_iirah = classification(T_iirah.U{1},G_iirah,X,labels); ta_iirah = cputime - tStart;
% % para_iirah = prod(size(T_iirah.core))+prod(size(T_iirah.U{1}))+prod(size(T_iirah.U{2}))+prod(size(T_iirah.U{3}));

% para_iirah = prod(size(T_iirah.core))+prod(size(T_iirah.U{1}))+prod(size(T_iirah.U{2}))+prod(size(T_iirah.U{3}))+prod(size(T_iirah.U{4}));

% % random rank-adaptive hooi
tStart = cputime; [T_prrah1,ranks_prrah1,~] = rand_rank_ada_hooi(A,epsilon,'rand','randQB','maxiters',iter); t_prrah1 = cputime - tStart;
% 
B_hat = tmprod(T_prrah1.core,T_prrah1.U,1:N);
res_prrah1 = frob(B_hat - B)/frob(B);
% 
% % G_prrah1 = tmprod(T_prrah1.core,T_prrah1.U{3},3);
% % ratio_prrah1 = prod(size(A))/prod(size(G_prrah1));
% % tStart = cputime; ca_prrah1 = classification(T_prrah1.U{1},G_prrah1,X,labels); ta_prrah1 = cputime - tStart;
% 
% para_prrah1 = prod(size(T_prrah1.core))+prod(size(T_prrah1.U{1}))+prod(size(T_prrah1.U{2}))+prod(size(T_prrah1.U{3}))+prod(size(T_prrah1.U{4}));
% 
% %tStart = cputime; [T_prrah2,ranks_prrah2,~] = rand_rank_ada_hooi(A,epsilon,'rand','randBGKL','maxiters',iter); t_prrah2 = cputime - tStart;
% 
% % B_hat = tmprod(T_prrah2.core,T_prrah2.U,1:N);
% % res_prrah2 = frob(B_hat - B)/frob(B);
% 
% % G_prrah2 = tmprod(T_prrah2.core,T_prrah2.U{3},3);
% % ratio_prrah2 = prod(size(A))/prod(size(G_prrah2));
% % tStart = cputime; ca_prrah2 = classification(T_prrah2.U{1},G_prrah2,X,labels); ta_prrah2 = cputime - tStart;

%para_prrah2 = prod(size(T_prrah2.core))+prod(size(T_prrah2.U{1}))+prod(size(T_prrah2.U{2}))+prod(size(T_prrah2.U{3}))+prod(size(T_prrah2.U{4}));

%%
subplot 121
plot(result(1,:),'-k')
hold on
plot(result(2,:),'-b')
hold on
plot(result(3,:),'-g')
hold on
plot(result(4,:),'-r')
hold on
plot(result(5,:),'-y')
hold on
plot(result(6,:),'-p')

subplot 122
bar(time')

%%

%normA = frob(A);

A_t = tmprod(G_t,U_t,1:4); %E_t = abs(A_t - A); E_t = E_t/normA;
A_st = tmprod(G_st,U_st,1:4); %E_st = abs(A_st - A); E_st = E_st/normA;
G_g = tmprod(A,T_g.U,1:4,'T'); A_g = tmprod(G_g,T_g.U,1:4); %E_g = abs(A_g - A); E_g = E_g/normA;
G_rah = tmprod(A,T_rah.U,1:4,'T'); A_rah = tmprod(G_rah,T_rah.U,1:4); %E_rah = abs(A_rah - A); E_rah = E_rah/normA;
G_prrah1 = tmprod(A,T_prrah1.U,1:4,'T'); A_prrah1 = tmprod(G_prrah1,T_prrah1.U,1:4); %E_prrah1 = abs(A_prrah1 - A); E_prrah1 = E_prrah1/normA;
%G_prrah2 = tmprod(A,T_prrah2.U,1:4,'T'); A_prrah2 = tmprod(G_prrah2,T_prrah2.U,1:4); %E_prrah2 = abs(A_prrah2 - A); E_prrah2 = E_prrah2/normA;

[X,Y] = meshgrid(-100:delta_x:100);

A_slice = A(51,51,:,:); A_slice = reshape(A_slice,[100,100]);
A_t_slice = A_t(51,51,:,:); A_t_slice = reshape(A_t_slice,[100,100]);
A_st_slice = A_st(51,51,:,:); A_st_slice = reshape(A_st_slice,[100,100]);
A_g_slice = A_g(51,51,:,:); A_g_slice = reshape(A_g_slice,[100,100]);
A_rah_slice = A_rah(51,51,:,:); A_rah_slice = reshape(A_rah_slice,[100,100]);
A_prrah1_slice = A_prrah1(51,51,:,:); A_prrah1_slice = reshape(A_prrah1_slice,[100,100]);
%A_prrah2_slice = A_prrah2(51,51,:,:); A_prrah2_slice = reshape(A_prrah2_slice,[100,100]);

subplot 231
pcolor(A_slice)
subplot 232
pcolor(A_t_slice)
subplot 233
pcolor(A_st_slice)
subplot 234
pcolor(A_g_slice)
subplot 235
pcolor(A_rah_slice)
subplot 236
pcolor(A_prrah1_slice)

% subplot 151
% pcolor(abs(A_t_slice - A_slice))
% subplot 152
% pcolor(abs(A_st_slice - A_slice))
% subplot 153
% pcolor(abs(A_g_slice - A_slice))
% subplot 154
% pcolor(abs(A_rah_slice - A_slice))
% subplot 155
% pcolor(abs(A_prrah1_slice - A_slice))

% num = 500;
% subplot 231
% contour(X,Y,A_slice,num)
% subplot 232
% contour(X,Y,A_t_slice,num)
% subplot 233
% contour(X,Y,A_st_slice,num)
% subplot 234
% contour(X,Y,A_g_slice,num)
% subplot 235
% contour(X,Y,A_rah_slice,num)
% subplot 236
% contour(X,Y,A_prrah1_slice,num)


% [X,Y,Z] = meshgrid(1:I);
% SIZE = 1:100;
% xslice = SIZE;
% yslice = SIZE;
% zslice = SIZE;

% A_slice = A(:,:,:,50);
% A_t_slice = A_t(:,:,:,50);
% A_st_slice = A_st(:,:,:,50);
% A_g_slice = A_g(:,:,:,50);
% A_rah_slice = A_rah(:,:,:,50);
% A_prrah1_slice = A_prrah1(:,:,:,50);
% A_prrah2_slice = A_prrah2(:,:,:,50);
% 
% subplot 231
% slice(X,Y,Z,A_t_slice,xslice,yslice,zslice);
% subplot 232
% slice(X,Y,Z,A_st_slice,xslice,yslice,zslice);
% subplot 233
% slice(X,Y,Z,A_g_slice,xslice,yslice,zslice);
% subplot 234
% slice(X,Y,Z,A_rah_slice,xslice,yslice,zslice);
% subplot 235
% slice(X,Y,Z,A_prrah1_slice,xslice,yslice,zslice);
% subplot 236
% slice(X,Y,Z,A_prrah2_slice,xslice,yslice,zslice);

% E_t_slice = E_t(:,:,:,50);
% E_st_slice = E_st(:,:,:,50);
% E_g_slice = E_g(:,:,:,50);
% E_rah_slice = E_rah(:,:,:,50);
% E_prrah1_slice = E_prrah1(:,:,:,50);
% E_prrah2_slice = E_prrah2(:,:,:,50);

% subplot 131
% slice(X,Y,Z,E_t_slice,xslice,yslice,zslice);
% subplot 132
% slice(X,Y,Z,E_st_slice,xslice,yslice,zslice);
% subplot 133
% slice(X,Y,Z,E_g_slice,xslice,yslice,zslice);

% subplot 131
% slice(X,Y,Z,E_rah_slice,xslice,yslice,zslice);
% subplot 132
% slice(X,Y,Z,E_prrah1_slice,xslice,yslice,zslice);
% subplot 133
% slice(X,Y,Z,E_prrah2_slice,xslice,yslice,zslice);

%%

clc

% m = 500; n = 50000;
% r = 100;
% U = rand(m,r); V = rand(n,r);
% A = U*V'; 
% E = randn(m,n); E = E/norm(E,'fro');
% delta = 1.0e-2;
% A = A + delta*E;

% A = rand(m,n);

sizeA = size(A);
m = sizeA(1); n = sizeA(2);

epsilon = 1.0e-2;

tStart = cputime; A_hat1 = exact_svd(A,epsilon); t_1 = cputime - tStart;
disp(A_hat1.rank);

tStart = cputime; A_hat2 = randQB_fa(A,epsilon,'power',true,'block',10); t_2 = cputime - tStart;
disp(A_hat2.rank);

tStart = cputime; A_hat3 = randBGKL_fa(A,epsilon,'block',10); t_3 = cputime - tStart;
disp(A_hat3.rank);

function A_hat = exact_svd(A,epsilon)

    normA = norm(A,'fro');

    [u,sigma,v] = svd(A,'econ');

    s_sum = 0.0;
    for i = length(diag(sigma)):-1:1
        s_sum = s_sum + sigma(i,i)^2;
        if s_sum >= epsilon^2*normA^2
            rank = i;
            break;
        end
    end

    A_hat.rank = rank;
    A_hat.U = u(:,1:rank);
    A_hat.sigma = sigma(1:rank,1:rank);
    A_hat.V = v(:,1:rank);

end

function ca = classification_1D(U,G,test,label)

% test is 784x10000
% label is 10000x1
% U is 784x65
% G is 65x142x10
k = 16;
for i = 1:10
  mat = reshape(G(:,:,i),[65,142]);
  [u,~,~] = svds(mat,k);
  cate{i} = u;
end
my_label = zeros(10000,1);
dis = zeros(1,10);
I = 1:10000;
for i = I
  x = test(:,i);
  x = U'*x;
  for j = 1:10
    M = cate{j};
    dis(j) = norm(x - M*M'*x);
  end
  class = find(dis == min(dis));
  my_label(i) = class - 1;
end
ca = find(my_label == label);
ca = length(ca)/10000;
end