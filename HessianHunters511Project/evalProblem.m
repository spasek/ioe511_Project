function [f, g, H] = evalProblem(problem, x)

switch problem.name

    case 'quadratic'
        A = problem.A; b = problem.b; c = problem.c;
        f = 0.5 * x' * A * x + b' * x + c;
        g = A * x + b;
        H = A;

    case 'rosenbrock'
        w = x(1); z = x(2);

        f = (1 - w)^2 + 100 * (z - w^2)^2;

        g = [
            -2*(1-w) - 400*w*(z - w^2);
             200*(z - w^2)
        ];

        H = [
            2 - 400*(z - w^2) + 800*w^2,   -400*w;
            -400*w,                        200
        ];

        case 'function2'
        w = x(1); z = x(2);
        y = problem.y;

        r = y - w*(1 - z.^((1:3)'));

        f = sum(r.^2);

        % gradient
        dw = -2 * sum(r .* (1 - z.^((1:3)')));
        dz =  2 * sum(r .* (w * (1:3)' .* z.^((1:3)'-1)));

        g = [dw; dz];

        % Hessian
        Hww = 2 * sum((1 - z.^((1:3)')).^2);

        Hzz = 2 * sum( (w*(1:3)'.*z.^((1:3)'-1)).^2 ...
            - r .* (w*(1:3)'.*((1:3)'-1).*z.^((1:3)'-2)) );

        Hwz = 2 * sum( -(1 - z.^((1:3)')).*(w*(1:3)'.*z.^((1:3)'-1)) ...
            + r.*((1:3)'.*z.^((1:3)'-1)) );

        H = [Hww, Hwz; Hwz, Hzz];


         case 'function3'
        z = x;
        n = problem.n;

        e = exp(z(1));

        f = (e - 1)/(e + 1) + 0.1*exp(-z(1)) + sum((z(2:end)-1).^4);

        g = zeros(n,1);
        g(1) = 2*e/(e+1)^2 - 0.1*exp(-z(1));
        g(2:end) = 4*(z(2:end)-1).^3;

        H = zeros(n,n);
        H(1,1) = 2*e*(1 - e)/(e+1)^3 + 0.1*exp(-z(1));
        H(2:end,2:end) = diag(12*(z(2:end)-1).^2);

    otherwise
        error('Unknown problem type')
end
end
