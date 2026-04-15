function [H_mod, eta] = modifiedNewtonHessian(H, beta)

H = (H + H') / 2;
d = diag(H);

if min(d) > 0
    eta = 0;
else
    eta = -min(d) + beta;
end

for k = 1:50
    [R, p] = chol(H + eta*eye(size(H)));
    if p == 0
        H_mod = H + eta*eye(size(H));
        return
    else
        eta = max(2*eta, beta);
    end
end

H_mod = H + eta*eye(size(H));
end
