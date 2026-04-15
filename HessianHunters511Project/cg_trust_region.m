function d = cg_trust_region(gk, Bk, Delta_k, eps_CG)
% One-function implementation of the CG Trust-Region Subproblem Solver
% Matches the pseudocode exactly.

% ---------------------------------------------------------
% [1] Initialization
% ---------------------------------------------------------
z = zeros(size(gk));
r = gk;
p = -r;

% ---------------------------------------------------------
% [2] Check initial residual
% ---------------------------------------------------------
if norm(r) < eps_CG
    d = z;
    return
end

% ---------------------------------------------------------
% [4] CG iterations
% ---------------------------------------------------------
while true

    % [5] Check curvature
    pBp = p' * (Bk * p);
    if pBp <= 0
        % [6] Find tau such that ||z + tau p|| = Delta_k
        tau = compute_tau(z, p, Delta_k);
        d = z + tau * p;
        return
    end

    % [8] Step length
    alpha = (r' * r) / pBp;
    z_next = z + alpha * p;

    % [9] Check trust-region boundary
    if norm(z_next) >= Delta_k
        % [10] Find tau such that ||z + tau p|| = Delta_k
        tau = compute_tau(z, p, Delta_k);
        d = z + tau * p;
        return
    end

    % [12] Update residual
    r_next = r + alpha * (Bk * p);

    % [13] Convergence check
    if norm(r_next) <= eps_CG
        d = z_next;
        return
    end

    % [15] Update beta and direction
    beta = (r_next' * r_next) / (r' * r);
    p = -r_next + beta * p;

    % Move to next iteration
    z = z_next;
    r = r_next;
end

% ---------------------------------------------------------
% Internal helper: tau for boundary intersection
% ---------------------------------------------------------
function tau = compute_tau(z, p, Delta)
    a = p' * p;
    b = 2 * (z' * p);
    c = z' * z - Delta^2;
    tau = (-b + sqrt(b^2 - 4*a*c)) / (2*a);
end

end
