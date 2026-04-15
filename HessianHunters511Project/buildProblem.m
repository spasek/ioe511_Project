function problem = buildProblem(type, varargin)
% buildProblem(type, ...)
% type = 'quadratic' or 'rosenbrock'
%
% Quadratic usage:
%   problem = buildProblem('quadratic', 'file', 'quadratic10.mat');
%
% Rosenbrock usage:
%   problem = buildProblem('rosenbrock', 'x0', [1.2;1.2]);

switch type

    % ---------------------------------------------------------
    % QUADRATIC (already implemented)
    % ---------------------------------------------------------
    case 'quadratic'
        p = inputParser;
        addParameter(p, 'file', '');
        parse(p, varargin{:});
        datafile = p.Results.file;

        if isempty(datafile)
            error('Quadratic problem requires a MAT file: use buildProblem(''quadratic'',''file'',''quadratic10.mat'')')
        end

        data = load(datafile);
        A = data.A; b = data.b; c = data.c;
        x0 = data.x_0; x_star = data.x_star;

        f_star = 0.5 * x_star' * A * x_star + b' * x_star + c;

        problem = struct( ...
            'name', 'quadratic', ...
            'A', A, 'b', b, 'c', c, ...
            'x0', x0, 'x_star', x_star, 'f_star', f_star );

    % ---------------------------------------------------------
    % ROSENBROCK (already implemented)
    % ---------------------------------------------------------
    case 'rosenbrock'
        p = inputParser;
        addParameter(p, 'x0', [1.2;1.2]);
        parse(p, varargin{:});
        x0 = p.Results.x0;

        problem = struct( ...
            'name', 'rosenbrock', ...
            'x0', x0, ...
            'x_star', [1;1], ...
            'f_star', 0 );

    % ---------------------------------------------------------
    % FUNCTION 2
    % f(x) = Σ_{i=1}^3 (y_i - w(1 - z^i))^2
    % ---------------------------------------------------------
    case 'function2'
        y = [1.5; 2.25; 2.625];
        x0 = [1; 1];

        problem = struct( ...
            'name', 'function2', ...
            'y', y, ...
            'x0', x0, ...
            'x_star', [], ...   % unknown closed form
            'f_star', [] );

    % ---------------------------------------------------------
    % FUNCTION 3
    % f(x) = (exp(z1)-1)/(exp(z1)+1) + 0.1 exp(-z1) + Σ_{i=2}^n (zi - 1)^4
    % ---------------------------------------------------------
    case 'function3'
        p = inputParser;
        addParameter(p, 'n', 10);
        parse(p, varargin{:});
        n = p.Results.n;

        x0 = [1; zeros(n-1,1)];

        problem = struct( ...
            'name', 'function3', ...
            'n', n, ...
            'x0', x0, ...
            'x_star', [], ...
            'f_star', [] );

    otherwise
        error('Unknown problem type: %s', type)
end

end
