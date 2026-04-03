function [x,xTrack,grad_norm,num_iter] = newtons_method(x0)
    x = x0; % Initial guess
    tol = 1e-10; % Tolerance
    max_iter = 10000; % Maximum iterations
    for iter = 1:max_iter
        [grad,hessian] = rosenbrock(x);
        step = - hessian \ grad; % Solve for step
        xTrack(iter,:) = x;
        grad_norm(iter) = norm(grad);
        x = x + step;
        num_iter = iter;
        if norm(grad) < tol
            break;
        end
    end
    
    disp('Newton Method Result:');
    disp(x);
    disp('Number of Iterations')
    disp(num_iter)

end


