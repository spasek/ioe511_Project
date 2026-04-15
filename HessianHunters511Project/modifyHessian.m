function H_mod = modifyHessian(H)
    % Example: shift eigenvalues
    [V, D] = eig((H + H')/2);  % symmetrize
    lambda_min = min(diag(D));
    delta = 0;
    if lambda_min <= 0
        delta = -lambda_min + 1e-6;
    end
    H_mod = H + delta * eye(size(H));
end
