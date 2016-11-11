function [G,iter,tau] = SP_FW(Kmax,mu,M,cst,dim,a,b,away,adaptive)

% Solve the saddle point problem
% L(x,y) = mu*\|x\|^2 - x^\top M y - mu*\|y\|^2
% Using the SF-FW algorithm see (Gidel, Jebara, Lacoste-Julien,2016)
% For more details. This is algorithm 2 of the paper if away = 0
% and algorithm 3 if away = 1.
% Each step does a FW step and record the value of the FW gap.
% Inputs:
%   Kmax: The number of iteration
%   mu: The value of the mu convexity constant
%   cst: The matrix M \sim Unif([-cst,cst]^2*dim)
%   dim: Dimension of the problem
%   a: position of the saddle point in \X
%   b: position of the saddle point in \Y
%   away: chose if the algorithm is SP-FW of SP-AFW
%   adaptive: define the step size of the algorithm
%
% %===== ALG CHOICE ========
    % use 0 for regular FW
    % use 1 for away step
% ==== STEP SIZE CHOICE =====
    % use adaptive == 1 for gap/C
    % use adaptive == 2 for extended line-search...
    % use adaptive == 3 for 2/(2+k)
    % use adaptive == 4 for gap/\tilde C (heuristic step size)
% ==== SAFETY =====
    % use safety == 1 for a step size max(2/2+k,step) (useful with extended line-search)
    % use safety == 0 for algorithm as described in the paper
safety = 0;

% ================= Definition of the LMO =======================
metaF = @(x,y,M,a,b,mu) [mu.*(x-a) - M*(y-b); M'*(x-a)+mu.*(y-b)];
F = @(x,y,M) metaF(x,y,M,a,b,mu); % F(z)
S = @(F) -(sign(F+eps)-1)*.5;

% ========== Computation of the constants of the problem ========
% we used the \ell_2 norm
delta = min(norm(min([1-a, a]'),2),norm(min([1-b,b]),2));
% \delta_\X :=  \min_{\s_x \in \partial  \X} \|\x^* -\s\|, \delta = \min (\delta_\x, \delta_\Y)
Pwidth = 1/sqrt(dim); %Cf On the global linear convergence of Frank-Wolfe optimization variants S Lacoste-Julien, M Jaggi - Advances in Neural Information …, 2015 - papers.nips.cc
diam = sqrt(dim); % computed with the \ell_2 norm
C_L = mu * diam^2; % exact computation of the curvature for quadratic objective
mu_A = mu * Pwidth^2;
mu_int = mu * delta^2 ;
M_L = sqrt(2/mu) * norm(M) * diam;
if away==1
    tau = 1/2 - M_L/(mu_A)
else
    tau = 1 - M_L/(mu_int)
end

% Maybe the "good" C 'Heuristic' step size
if adaptive == 4
    C = 2*mu*diam^2+ norm(M)^2*(diam^2/mu+diam^2/mu)
else
    C = 2*mu*diam^2/tau;
end

if adaptive > 0
    gamma_coef = 1/(C);
end

% ==== INITIALIZATION ====
x_0 = [ones(dim,1); zeros(dim,1)];
S_0 = x_0;
alpha_0 = 1;

iter = 1;
tol = 1e-6;
G = [];

z           = x_0;
S_t         = S_0;
alpha_t     = alpha_0;

% each column of S_t is a potential vertex
% I_active -> contains index of active vertices (same as alpha_t > 0)
mapping = containers.Map();
% this will map vertex hash to id in S_t (to see whether already seen vertex)
% alpha_t(i) == 0 implies vertex is not active anymore...
% alpha_t will be a weight vector so that x_t = S_t * alpha_t

% constructing mapping:
max_index = size(S_t,2); % keep track of size of S_t
for index = 1:max_index
    mapping(hashing(S_t(:,index))) = index;
end
I_active = find(alpha_t > 0);
gap_best = 10^10;

% tracking results:
number_drop = 0; % counting drop steps (max stepsize for away step)
number_away = 0;

% ==== MAIN ALGORITHM ====
for k = 0:Kmax
    Fz = F(z(1:dim),z(dim+1:end),M);
    Sz = S(Fz); % FW-corner
    s_FW = Sz;
    grad = Fz;
    d_FW     = s_FW - z;
    gap= - d_FW' * grad;
    gap_best = min(gap,gap_best);
    G(iter) = gap_best; % keep track of FW gaps..
    if gap < tol
        fprintf('Converged at iter %d!\n', iter)
        if gap < 0
            assert false
        end
        break
    end
    if away
        % away direction search:
        if ~isempty(S_t)
            id_A   = away_step(Fz, S_t, I_active);
            v_A    = S_t(:, id_A);
            d_A    = z - v_A;
            alpha_max = alpha_t(id_A);
        else
            fprintf('error: empty support set at step (it=%f)\n', it);
        end

        % construct direction (between towards and away):
        if isempty(S_t) || - gap <= d_A' * grad
            is_aw = false; % this is a pure FW step
            d = d_FW;
            max_step = 1;
        else
            is_aw = true; % this is an away step
            number_away = number_away+1;
            d = d_A;
            max_step = alpha_max / (1 - alpha_max);
        end

        gap_AFW = -Fz'*d;
        if safety
            gamma_max = 2/(k+2);
        else
            gamma_max = 1;
        end
        gamma = gamma_coef*gap_AFW;
        gamma_k = min([gamma,gamma_max,max_step]);
        if adaptive == 3
            gamma_k = min(2./(2+iter),max_step);
        end
        step = gamma_k;

        % doing steps and updating active set:

        if is_aw
            % away step:
            alpha_t = (1+step)*alpha_t; % note that inactive should stay at 0;
            if abs(step - max_step) < 10*eps
                % drop step:
                number_drop = number_drop+1;
                alpha_t(id_A) = 0;
                I_active(I_active == id_A) = []; % remove from active set
            else
                alpha_t(id_A) = alpha_t(id_A) - step;
            end
        else
            % FW step:
            alpha_t = (1-step)*alpha_t;

            % is this s_FW a new vertex?
            h = hashing(2*s_FW-1);
            if ~mapping.isKey(h)
                % we need to add vertex in S_t:
                max_index = max_index + 1;
                mapping(h) = max_index;
                S_t(:,max_index) = s_FW;
                id_FW = max_index;
                alpha_t(id_FW) = step; % this increase size of alpha_t btw
                I_active = [I_active, id_FW];
            else
                id_FW = mapping(h);
                if alpha_t(id_FW) < eps
                    % we already had atom in 'correction poytope', but it was not
                    % active, so now track it as active:
                    I_active = [I_active, id_FW];
                end
                alpha_t(id_FW) = alpha_t(id_FW) + step;
            end
            % exceptional case: stepsize of 1, this collapses the active set!
            if step > 1-eps;
                I_active = [id_FW];
            end
        end
    elseif adaptive ==3
            step = 2./(2+iter);
            d = d_FW;
    else

            step = min(1,gamma_coef*gap);
            d = d_FW;
    end
    z = z + step * d;
    iter = iter+1;
end
