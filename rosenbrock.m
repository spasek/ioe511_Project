function [grad,hessian,fval] = rosenbrock(x)
    grad = zeros(size(x));
    grad(1) = 2 * (x(1) - 1) - 400 * x(1) * (x(2) - x(1)^2);
    grad(2) = 200 * (x(2) - x(1)^2);

     hessian = zeros(length(x));
    hessian(1, 1) = 2 - 400 * (x(2) - 3 * x(1)^2);
    hessian(1, 2) = -400 * x(1);
    hessian(2, 1) = -400 * x(1);
    hessian(2, 2) = 200;

    fval = (1-x(1))^2+100*(x(2)-x(1)^2)^2;
end
