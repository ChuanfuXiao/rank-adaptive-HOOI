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

