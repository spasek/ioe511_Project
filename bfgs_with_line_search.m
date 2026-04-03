function  [x,xTrack,grad_norm,num_iter] = bfgs_with_line_search(x0)
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
        
        p = -H * grad; % Compute direction

        if dot(grad,p) > 0
            p = -p;
        end

        lambda = backtracking_line_search(@rosenbrock, x, p);
        s = lambda * p;
        x_new = x + s;
        [new_grad,~,~] = rosenbrock(x_new);
        y = new_grad - grad;
        rho = 1 / (y' * s);
        
        H = (eye(n) - rho * (s * y')) * H * (eye(n) - rho * (y * s')) + rho * (s * s');
        x = x_new;
       
    end
    
    disp('BFGS With Line Search Result:');
    disp(x);
    disp('Number of Iterations')
    disp(num_iter)
end

function lambda = backtracking_line_search(func, x, p)
    [grad, ~, fval] = func(x); mu = 1e-5; beta = 0.25;
    lambda = 1;
     [~,~,phi] = func(x+lambda*p);
   
    while (phi) > fval + mu*lambda*grad'*p
       lambda = beta*lambda;
       [~,~,phi] = func(x+lambda*p);
    end
end

