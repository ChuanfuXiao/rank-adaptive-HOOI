count = 100;

epsilon = 1.0e-2;

para = zeros([count,1]);
para_t = zeros([count,1]);
para_st = zeros([count,1]);
para_g = zeros([count,1]);
para_r1 = zeros([count,1]);
para_r2 = zeros([count,1]);

t_t = zeros([count,1]);
t_st = zeros([count,1]);
t_g = zeros([count,1]);
t_r1 = zeros([count,1]);
t_r2 = zeros([count,1]);

delta_r1 = 5*ones(1,4);
delta_r2 = 10*ones(1,4);

for t = 1:count
    sizeA = randi([50,150],1,4);
    delta_x= 200./(sizeA-1);
    x1 = -100:delta_x(1):100;
    x2 = -100:delta_x(2):100;
    x3 = -100:delta_x(3):100;
    x4 = -100:delta_x(4):100;
    
    A = zeros(sizeA);
    for i=1:sizeA(1)
        for j = 1:sizeA(2)
            for k = 1:sizeA(3)
                for l = 1:sizeA(4)
                    A(i,j,k,l) = log(0.1+sqrt(abs(x1(i)-x2(j))^2)+sqrt(abs(x3(k)-x4(l))^2));
                end
            end
        end
    end

    para(t) = prod(sizeA);

    tstart = cputime; [U_t,G_t] = mlsvd(A,epsilon,0); t_t(t) = cputime - tstart;
    inter_para = prod(size(G_t))+prod(size(U_t{1}))+prod(size(U_t{2}))+prod(size(U_t{3}))+prod(size(U_t{4}));
    para_t(t) = inter_para;

    tstart = cputime; [U_st,G_st] = mlsvd(A,epsilon); t_st(t) = cputime - tstart;
    ratio_st(t,:) = size(G_st)./sizeA;
    inter_para = prod(size(G_st))+prod(size(U_st{1}))+prod(size(U_st{2}))+prod(size(U_st{3}))+prod(size(U_st{4}));
    para_st(t) = inter_para;

    tstart = cputime; [T_g,ranks_g] = greedy_hosvd(A,epsilon); t_g(t) = cputime - tstart;
    ratio_g(t,:) = ranks_g'./sizeA;
    inter_para = prod(size(T_g.core))+prod(size(T_g.U{1}))+prod(size(T_g.U{2}))+prod(size(T_g.U{3}))+prod(size(T_g.U{4}));
    para_g(t) = inter_para;

    tstart = cputime; [T_rah1,ranks_rah1,iter1] = rank_ada_hooi(A,epsilon,'delta_r',delta_r1); t_r1(t) = cputime - tstart;
    ratio_r1(t,:) = ranks_rah1./sizeA;
    inter_para = prod(size(T_rah1.core))+prod(size(T_rah1.U{1}))+prod(size(T_rah1.U{2}))+prod(size(T_rah1.U{3}))+prod(size(T_rah1.U{4}));
    para_r1(t) = inter_para;

    tstart = cputime; [T_rah2,ranks_rah2,iter2] = rank_ada_hooi(A,epsilon,'delta_r',delta_r2); t_r2(t) = cputime - tstart;
    ratio_r2(t,:) = ranks_rah2./sizeA;
    inter_para = prod(size(T_rah2.core))+prod(size(T_rah2.U{1}))+prod(size(T_rah2.U{2}))+prod(size(T_rah2.U{3}))+prod(size(T_rah2.U{4}));
    para_r2(t) = inter_para;
end

