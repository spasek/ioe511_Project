% Colton Knowles
% IOE 511 Final Project
%
% Strong Wolfe line search (Nocedal & Wright, Algorithms 3.5 & 3.6)
%
% Two stages:
%   1. Bracket: increase alpha until the interval [alpha_prev, alpha]
%               must contain a good step (Algorithm 3.5)
%   2. Zoom:    shrink that interval until both Wolfe conditions hold
%               (Algorithm 3.6)
%
% Wolfe conditions:
%   Sufficient decrease:  f(x + a*d) <= f(x) + c1*a*(g'd)   [Armijo]
%   Strong curvature:    |g(x + a*d)'*d| <= c2*|g(x)'*d|

function [alpha, f_new, g_new] = wolfeLineSearch(x, f, g, d, problem, options)

c1        = options.c1;
c2        = options.c2;
alpha_max = options.alpha_bar;
max_iters = 50;

% phi(alpha) = f(x + alpha*d), phi'(alpha) = g(x + alpha*d)'*d
phi0  = f;
dphi0 = g' * d;   % must be negative (descent direction)

if dphi0 >= 0
    error('wolfeLineSearch: search direction is not a descent direction.');
end

% Initialize: previous step = 0, first trial step = 1
alpha_prev = 0;
f_prev     = phi0;
alpha      = min(1, alpha_max);

% Stage 1: Bracket (Algorithm 3.5)
for i = 1:max_iters

    f_trial = problem.compute_f(x + alpha*d);
    feval_counter('f');

    % If Armijo fails or phi increased vs. last step, a minimum is
    % trapped in [alpha_prev, alpha]. Therefore, zoom in on it
    if (f_trial > phi0 + c1*alpha*dphi0) || (i > 1 && f_trial >= f_prev)
        [alpha, f_new, g_new] = zoom(alpha_prev, alpha, x, phi0, dphi0, d, problem, options);
        return;
    end

    % Armijo holds: check the curvature condition
    t0       = tic;
    g_trial  = problem.compute_g(x + alpha*d);
    timing_counter('grad', toc(t0));
    feval_counter('g');
    dphi_cur = g_trial' * d;   % phi'(alpha)

    % Both Wolfe conditions satisfied: accept this step
    if abs(dphi_cur) <= c2 * abs(dphi0)
        f_new = f_trial;
        g_new = g_trial;
        return;
    end

    % Slope is positive: minimum is between alpha_prev and alpha,
    % but now alpha is the better endpoint
    if dphi_cur >= 0
        [alpha, f_new, g_new] = zoom(alpha, alpha_prev, x, phi0, dphi0, d, problem, options);
        return;
    end

    % Still descending steeply: take a larger step
    alpha_prev = alpha;
    f_prev     = f_trial;
    alpha      = min(2*alpha, alpha_max);

end

% Fallback if max iterations reached
f_new = problem.compute_f(x + alpha*d);
t0    = tic;
g_new = problem.compute_g(x + alpha*d);
timing_counter('grad', toc(t0));
feval_counter('f');
feval_counter('g');

end
