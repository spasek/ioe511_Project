function [x,xTrack,grad_norm,num_iter] =bfgs_without_line_search(x0)
    x = x0; % Initial guess
    tol = 1e-10; % Tolerance
    max_iter = 1000; % Maximum iterations
    n = length(x);
    H = eye(n); % Approximate Hessian matrix

    for iter = 1:max_iter
        [grad,~] = rosenbrock(x);
        num_iter = iter;
        xTrack(iter,:) = x;
        grad_norm(iter) = norm(grad);
        if norm(grad) < tol
            break;
        end
        
        step = -H * grad; % Compute step
        x_new = x + step;
        s = x_new - x;
        y = rosenbrock(x_new) - grad;
        rho = 1 / (y' * s);
        
        if dot(grad,step) > 0
            step = -step;
        end

        H = (eye(n) - rho * (s * y')) * H * (eye(n) - rho * (y * s')) + rho * (s * s');
        x = x_new;
        
    end
    
    disp('BFGS Without Line Search Result:');
    disp(x);
    disp('Number of Iterations')
    disp(num_iter)
end

