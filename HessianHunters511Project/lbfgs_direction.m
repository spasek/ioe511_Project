function d = lbfgs_direction(gk, S, Y)
% L-BFGS two-loop recursion, H0 = I

m = length(S);
q = gk;
alpha = zeros(m,1);
rho   = zeros(m,1);

% First loop (from most recent to oldest)
for i = 1:m
    s = S{i};
    y = Y{i};
    rho(i)   = 1 / (y' * s);
    alpha(i) = rho(i) * (s' * q);
    q        = q - alpha(i) * y;
end

% H0 = I
r = q;

% Second loop (reverse order)
for i = m:-1:1
    s = S{i};
    y = Y{i};
    beta = rho(i) * (y' * r);
    r    = r + s * (alpha(i) - beta);
end

d = r;   % this is H_k * gk; caller uses d = -d
end
