function pval = legeval(C,x)

%% LEGEVAL.m Evaluate polynomial in the Legendre basis
%
% p = legeval(C,x) evaluates at x a polynomial in the Legendre basis, with
% coefficients specified by C. The entry C(i) is the coefficient of the
% Legendre polynomial of degree i-1.
%
% ----------------------------------------------------------------------- %
%        Author:    Giovanni Fantuzzi
%                   Department of Aeronautics
%                   Imperial College London
%       Created:    09/10/209
% Last Modified:    09/10/2019
% ----------------------------------------------------------------------- %

N = size(C,1);        % number of Legendre coefficients
Pnm1= zeros(size(x)); % dummy
Pn = ones(size(x));   % P0
pval = Pn*C(1,:);     % Initial function evaluation

% If have more than 1 coefficient
for n = 0:N-2
    Pnp1 = ( ((2*n+1)/(n+1)).*x ).*Pn - (n/(n+1)).*Pnm1;  % 3 term recurrence
    pval = pval + Pnp1*C(n+2,:);                  % Add to function evaluation
    Pnm1= Pn;
    Pn = Pnp1;
end