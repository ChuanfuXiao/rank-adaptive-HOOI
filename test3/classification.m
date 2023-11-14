function ca = classification(U,G,test,label)

% test is 784 x 10000
% label is 10000 x 1
% U is 784 x 65
% G is m x n x 10
k = 16;
sizeG = size(G);
m = sizeG(1); n = sizeG(2);

for i = 1:10
  mat = reshape(G(:,:,i),[m,n]);
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