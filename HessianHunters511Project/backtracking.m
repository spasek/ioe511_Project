function alpha = backtracking(problem, xk, fk, gk, dk, alpha_bar, c1, tau)
alpha = alpha_bar;

while true
    x_trial = xk + alpha * dk;
    [f_trial, ~, ~] = evalProblem(problem, x_trial);

    if f_trial <= fk + c1 * alpha * (gk' * dk)
        break;
    end

    alpha = tau * alpha;
end
end
