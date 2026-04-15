function [xk, fk, history] = optSolver_Pasek_Sean(problem, method, options)
% Main optimization solver

tic;
xk        = problem.x0(:);
alpha_bar = options.alpha_bar;
c1        = options.c1;
tau       = options.tau;
max_iters = options.max_iters;
tol       = options.tol;

history.f_vals      = [];
history.grad_norms  = [];
history.alphas      = [];
history.time        = 0;

Hk = eye(length(xk));   % for BFGS
S  = {};                % for L-BFGS
Y  = {};

for k = 1:max_iters
    [f, g, H] = evalProblem(problem, xk);

    history.f_vals(end+1)     = f;
    history.grad_norms(end+1) = norm(g, inf);

    % termination condition
    if k == 1
        grad0_inf = max(norm(g, inf), 1);
    end
    if norm(g, inf) <= grad0_inf * tol
        break
    end

    if k == 1
        Hk = eye(length(xk));
    end

switch method.type
    % ==================================================
    % GRADIENT DESCENT VARIANTS
    % ==================================================
    case 'gd_bt'
        d = -g;
        alpha = backtracking(problem, xk, f, g, d, alpha_bar, c1, tau);
        
    case 'gd_wolfe'
        d = -g;
        alpha = wolfeLineSearch(problem, xk, f, g, d, method.params);

    % ==================================================
    % MODIFIED NEWTON VARIANTS
    % ==================================================
    case 'newton_mod_bt'
        H = (H + H') / 2; % Ensure symmetry
        [H_mod, ~] = modifiedNewtonHessian(H, 1e-6);
        d = -H_mod \ g;
        alpha = backtracking(problem, xk, f, g, d, alpha_bar, c1, tau);

    case 'newton_mod_wolfe'
        H = (H + H') / 2;
        [H_mod, ~] = modifiedNewtonHessian(H, 1e-6);
        d = -H_mod \ g;
        alpha = wolfeLineSearch(problem, xk, f, g, d, method.params);

    case 'cg_tr_newton'
        % Conjugate Gradient Trust Region (Steihaug)
        H = (H + H') / 2;
        [H_mod, ~] = modifiedNewtonHessian(H, 1e-6);
        [s, tr_info] = trust_region_subproblem(g, H_mod, delta, 'cg');
        [xk, f, g, delta] = update_tr_step(problem, xk, f, g, s, delta);

    % ==================================================
    % BFGS VARIANTS
    % ==================================================
    case 'bfgs_bt'
        d = -Hk * g;
        if g' * d >= 0, d = -g; end 
        alpha = backtracking(problem, xk, f, g, d, alpha_bar, c1, tau);
        [xk, f, g, Hk] = update_quasi_newton(problem, xk, f, g, d, alpha, Hk, 'BFGS');

    case 'bfgs_wolfe'
        d = -Hk * g;
        if g' * d >= 0, d = -g; end
        alpha = wolfe_line_search(problem, xk, f, g, d, method.params);
        [xk, f, g, Hk] = update_quasi_newton(problem, xk, f, g, d, alpha, Hk, 'BFGS');

    % ==================================================
    % DFP VARIANTS
    % ==================================================
    case 'dfp_bt'
        d = -Hk * g;
        if g' * d >= 0, d = -g; end
        alpha = backtracking(problem, xk, f, g, d, alpha_bar, c1, tau);
        [xk, f, g, Hk] = update_quasi_newton(problem, xk, f, g, d, alpha, Hk, 'DFP');

    case 'dfp_wolfe'
        d = -Hk * g;
        if g' * d >= 0, d = -g; end
        alpha = wolfe_line_search(problem, xk, f, g, d, method.params);
        [xk, f, g, Hk] = update_quasi_newton(problem, xk, f, g, d, alpha, Hk, 'DFP');

    % ==================================================
    % SR1 VARIANTS
    % ==================================================
    case 'sr1_tr'
        % Symmetric Rank 1 Update with Trust Region
        [s, tr_info] = trust_region_subproblem(g, Hk, delta, 'exact');
        [xk, f, g, delta, Hk] = update_tr_sr1(problem, xk, f, g, s, delta, Hk);

    otherwise
        error('Method type "%s" not recognized.', method.type);
end

    history.alphas(end+1) = alpha;

    % For methods that didn't already update xk inside the case
    if ~ismember(method.type, {'bfgs','lbfgs'})
        xk = xk + alpha * d;
    end
end

[fk, ~, ~] = evalProblem(problem, xk);
history.time = toc;
end
