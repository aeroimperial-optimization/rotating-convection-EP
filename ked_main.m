% main.m
%
% Compute upper bound on the mean vertical heat transport in the KED reduced
% model of rotating convection. Assume infinite Prantl nuumber.
%
% This code implements the "outer approximation" of the variational problem
% for the bound described in this paper:
%
%   G. Fantuzzi, A. Wynn, A. Papachristodoulou and P. J. Goulart. 
%   "Optimization with affine homogeneous quadratic integral inequality 
%   constraints". IEEE Trans. Autom. Control 62(12):6221–6236.
%   https://doi.org/10.1017/jfm.2017.858 [Open Access]
% 
% The method is implemented in QUINOPT, but here we implement it manually
% to explicitly invert the equation for the vertical velocity w and express
% it in terms of T, rather than write T as a function of w. This
% significantly improves numerical conditioning (but makes this code a
% little harder to understand).
%
% NOTES
% -----
% (1) This code works on the rescaled interval [-1,1] for convenience
% (3) For a faster version of the function "legendreTripleProduct", copy 
%     and compile the relevant .mex file from QUINOPT:
%     https://github.com/aeroimperial-optimization/QUINOPT/blob/master/
%     utils/LegendreExpansion/private/computeTripleProducts.c
% (2) This code depends on the following toolboxes:
%     * YALMIP: https://yalmip.github.io/
%     * MOSEK: https://www.mosek.com/

% ----------------------------------------------------------------------- %
% Values of ee, R for sensible results:
% E = 1e-5, Ra =  1e9: OK with deg_phi_prime = 200, Nk = 30
% E = 1e-6, Ra = 1e10: OK with deg_phi_prime = 300, NK = 20
% E = 1e-7, Ra = 2e11: OK with deg_phi_prime = 400, Nk = 15
% E = 1e-9, Ra = 6e13: OK with deg_phi_prime = 600, Nk = 15

% ----------------------------------------------------------------------- %
% Start
clear
yalmip clear
close all
tstart = tic;

% ----------------------------------------------------------------------- %
% Parameters:
% * E: Eckman number
% * Ra: Rayleigh number
% * Lx: horizontal period (take large to approximate infinite layer)
% * Nk: number of horizontal wavenumbers for which to impose the constraints
% * deg_phi_prime: degree of polynomial ansatz for \phi'(z). Should be EVEN
% * NN: number of Legendre expansion terms in the test functions T and w.
%       It should be larger than deg_phi_prime (determine using a
%       convergence study)
E = 1e-5;
Ra = 7e7;
deg_phi_prime = 100;  
NN = 150; 
Lx = 50;
Nk = 10;

% ----------------------------------------------------------------------- %
% Scaled parameters and wavenumbers
ee = E^(1/3);                                 % Scaled Eckmann number
R  = Ra*E^(4/3);                              % Scaled Rayleigh number
kvals = 2*pi*(1:Nk)'./Lx;                     % List of wavenumbers

% ----------------------------------------------------------------------- %
% Create optimization variables:
% * U: the bound (scalar)
% * b: the "balance parameter"
% * Dphi_coeffs: the coefficients of an odd polynomial with zero mean,
%                expressed in the Legendre basis. To have zero mean, simply
%                set to zero the first coefficient because nonconstant 
%                Legendre polynomials have zero mean!
sdpvar U b
d = 0.5*deg_phi_prime+1;                   % number of tunable coefficients;
DphiC = sdpvar(d-1,1);                     % the tunable variables;
Dphi_coeffs = [0, DphiC'; zeros(1,d)];     % all coefficients: zero for
Dphi_coeffs = Dphi_coeffs(1:end-1)';       % Legendre polynomials of even degree

% ----------------------------------------------------------------------- %
% Expansion of S0 -- this is simple
% Enforce the boundary condition \theta(1)=0 explicitly, the other is 
% automatically enforced by dropping the integration term in the Legendre
% series expansions. By symmetry, can assume that \theta(z) is odd.
% NOTE: must account for rescaling of the Legendre expansions for
%       compatibility with the function "legendreDiff" from QUINOPT.
ll = Dphi_coeffs.*sqrt( 2./(1+2.*(0:deg_phi_prime)') );
S0 = [U-1, -ll.'; -ll, 2*(b-1)*eye(deg_phi_prime+1)];
S0 = S0([1,3,4:2:end],[1,3,4:2:end]);

% ----------------------------------------------------------------------- %
% Set the first constraint: S0 is nonnegative
F = [ S0>=0 ];

% ----------------------------------------------------------------------- %
% Expansion of Sk -- this is more complicated
% Step (1) : get some useful integration matrices (we expand the first
% derivative of T and the second derivative of w using Legendre series, 
% and recover the expansions of T and w by integration using the boundary
% conditions)
Sk = cell(Nk,1);
[D0w, B0w] = legendreDiff(NN,0,2,2);
[D1w, B1w] = legendreDiff(NN,0,1,2);
[D2w, B2w] = legendreDiff(NN,0,0,2);
D0w = D0w(1:end-4,1:end-4); B0w = B0w(1:end-4,:);
D1w = D1w(1:end-2,1:end-4); B1w = B1w(1:end-2,:);
D2w = D2w(:,1:end-4);
D1T = legendreDiff(NN,0,0,1);
D1T = D1T(:,1:end-2);
Z = sparse(2,NN);
MT = D1T.';
Mw = [B2w, D2w];

% Step (2): Compute the product of three Legendre polynomials. These arise 
% from the expansion of (b-2*Dphi)*w*theta in Sk. The factor of 2 comes from
% rescaling to [-1,1].
pp = [b; -2.*Dphi_coeffs(2:end)];           % Leg. coeffs of b-\phi'
nnzIdx = find(any(pp));
pp = pp(nnzIdx);
nnzpp = length(pp);
X = legendreTripleProduct(nnzIdx-1,0,NN,0,NN);
for j = 1:nnzpp
    % Multiply on the left by the integration matrix for T, so the
    % quadratic form is written (implicity) using the Legendre coefficients
    % of the derivative T'.
    X{j} = MT*X{j}; 
end

% Step (3): Build each Sk by looping over wavenumbers
DTMATRIX = (b-1).*((4.*ee).*eye(NN));   % expansion of (b-1)*||T_k'||^2
THETASQ = D1T.'*D1T;                    % expansion of ||T_k||^2
vars = depends(pp);                     % ID of optimization variables
for i = 1:Nk
    
    fprintf('Setting up wavenumber %i\n',i)
    k = kvals(i);
    
    % Step (3a): Express the coefficients of w_k''(z) as a function of the
    %            Legendre coefficients of T_k'(z), by solving the equation
    %            linking temperature and velocity.
    BC1 = [k^2, -sqrt(2*ee), sparse(1,NN-1)];
    BC2 = k^2.*( sqrt(2).*[B1w(1,:), D1w(1,:)] + [1, sparse(1,NN)]);
    BC2(2:3) = BC2(2:3) + [1, sqrt(2)]*sqrt(2*ee);
    A = k^2.*Mw(1:end-2,:) - (4/k^4).*[B0w, D0w];
    A = [BC1; BC2; A];
    B = [Z; D1T(1:end-2,:)];
    C = A\B;
    C = clean(C,1e-14);     % Clean up roundoff errors that ruin sparsity?
    
    % Step (3b): matrix representation of the term (b-Dphi)*theta*w. Use
    %            low-level commands to avoid expensive multiplication of
    %            symbolic variables. It's still not fast, but hey...
    Q = Mw*C;
    temp = cellfun(@(A)A*Q,X,'UniformOutput',0);
    temp(2:end) = cellfun(@(A)-2.*A,temp(2:end),'UniformOutput',0);
    temp = cellfun(@(A)A(:),temp,'UniformOutput',0);
    basis = (R/ee).*[sparse(NN^2,1), temp{:}];
    Q = sdpvar(NN,NN,[],vars,basis);
    
    % Step (3c): Assemble all terms contributing to Sk
    Sk{i} = DTMATRIX;                                   % eps^2*|theta'|^2
    Sk{i} = Sk{i} + (b-1).*((k^2/ee).*THETASQ);         % k^2*|theta|^2
    Sk{i} = Sk{i} - Q;                                  % (b-Dphi)*theta*w
    
    % Step (3d): Enforce the boundary condition \theta(1)=0. The first 
    %            expansion coefficient vanishes, so remove the first row
    %            and column from Sk (like in usual spectral methods)
    Sk{i} = Sk{i}(2:end,2:end);
    
    % Step (3e): Symmetrize and add the constraint that Sk be nonnegative
    Sk{i} = (Sk{i} + Sk{i}.');
    F = F + [Sk{i}>=0];
    
end

% ----------------------------------------------------------------------- %
% Optimize using YALMIP and MOSEK
fprintf('Solving...')
opts = sdpsettings('solver','mosek','verbose',1);
sol = optimize(F,U,opts);
check(F)
runtime = toc(tstart);

% ----------------------------------------------------------------------- %
% Extract the optimal solution and plot the optimal \phi(z)
Dphi_coeffs = value(Dphi_coeffs);
b = value(b);
zz = 0.5*(1-cos(linspace(0,pi,1e3))');
Dphi = 2.*legeval(Dphi_coeffs,2*zz-1);
plot(zz,cumtrapz(zz,Dphi),'-','linewidth',1)

% ----------------------------------------------------------------------- %
% Finally, check that Sk>=0 for more wavenumbers than considered in the
% optimisation. If this is not true, the solution is invalid and should be
% recomputed including more wavenumbers. Here, no optimization variables
kcheck = 2*pi*(1:300)'./Lx;
pp = [b; -2.*Dphi_coeffs(2:end)];
nnzIdx = find(abs(pp)>0);
pp = pp(nnzIdx);
nnzpp = length(pp);
X = legendreTripleProduct(nnzIdx-1,0,NN,0,NN);
Q = pp(1).*X{1};
for j = 2:nnzpp, Q = Q + pp(j).*X{j}; end
Q = (R/ee).*MT*Q;
DTMATRIX = (b-1).*((4.*ee).*eye(NN));

% Loop over all wavenumbers
EV = zeros(size(kcheck));
fprintf('Checking more wavenumbers...\n')
for i = 1:length(kcheck)
    
    k = kcheck(i);
    
    % Solve for the legendre coefficients of w_k''(z) as a function of the
    % Legendre coefficients of T_k'(z), by solving the equation linking 
    % temperature and velocity at a given wavenumber. (SAME AS ABOVE)
    BC1 = [k^2, -sqrt(2*ee), sparse(1,NN-1)];
    BC2 = k^2.*( sqrt(2).*[B1w(1,:), D1w(1,:)] + [1, sparse(1,NN)]);
    BC2(2:3) = BC2(2:3) + [1, sqrt(2)]*sqrt(2*ee);
    A = k^2.*Mw(1:end-2,:) - (4/k^4).*[B0w, D0w];
    A = [BC1; BC2; A];
    B = [Z; D1T(1:end-2,:)];
    C = A\B;
    C = clean(C,1e-14);
    
    % Build the quadratic form Sk
    Sk_check = DTMATRIX;                                    % eps^2*|theta'|^2
    Sk_check = Sk_check + ((b-1)*k^2/ee).*THETASQ;          % k^2*|theta|^2
    Sk_check = Sk_check - Q*(Mw*C);                         % (b-Dphi)*theta*w
    Sk_check = Sk_check(2:end,2:end);                       % enforce BCs
    Sk_check = Sk_check + Sk_check.';                       % Symmetrize
    
    % Check positive semidefiniteness
    % (could use LDL factorization instead of eig, to check for positivity,
    % but would not get the trend of ground-state eigenvalues)
    EV(i,1) = min(eig(full(Sk_check)));
    
end
