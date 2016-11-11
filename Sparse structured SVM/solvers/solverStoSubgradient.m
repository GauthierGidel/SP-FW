function [model, progress] = solverSPFW(param, options)
% [model, progress] = solverFW(param, options)
%
% Solves the structured support vector machine (SVM) using Stochastic subgradient
% descent (SSG), see (Gidel, Jebara, Lacoste Julien, 2016) for more details.
% This is Algorithm 2 in the paper, and the code here follows a similar
% notation. Each step of the method calls the decoding oracle (i.e. the
% loss-augmented predicion) for all points.
%
% All our code is largely inspired from the code of BCFW algorithm
% [BCFW](https://github.com/ppletscher/BCFWstruct)
% see also (Lacoste-Julien, Jaggi, Schmidt, Pletscher; ICML
% 2013)
% Inputs:
%   param: a structure describing the problem with the following fields:
%
%     patterns  -- patterns (x_i)
%         A cell array of patterns (x_i). The entries can have any
%         nature (they can just be indexes of the actual data for
%         example).
%
%     labels    -- labels (y_i)
%         A cell array of labels (y_i). The entries can have any nature.
%
%     lossFn    -- loss function callback
%         A handle to the loss function L(ytruth, ypredict) defined for
%         your problem. This function should have a signature of the form:
%           scalar_output = loss(param, ytruth, ypredict)
%         It will be given an input ytruth, a ground truth label;
%         ypredict, a prediction label; and param, the same structure
%         passed to solverFW.
%
%     oracleFn  -- loss-augmented decoding callback
%         [Can also be called constraintFn for backward compatibility with
%          code using svm_struct_learn.]
%         A handle to the 'maximization oracle' (equation (2) in paper)
%         which solves the loss-augmented decoding problem. This function
%         should have a signature of the form:
%           ypredict = decode(param, model, x, y)
%         where x is an input pattern, y is its ground truth label,
%         param is the input param structure to solverFW and model is the
%         current model structure (the main field is model.w which contains
%         the parameter vector).
%
%     featureFn  feature map callback
%         A handle to the feature map function \phi(x,y). This function
%         should have a signature of the form:
%           phi_vector = feature(param, x, y)
%         where x is an input pattern, y is an input label, and param
%         is the usual input param structure. The output should be a vector
%         of *fixed* dimension d which is the same
%         across all calls to the function. The parameter vector w will
%         have the same dimension as this feature map. In our current
%         implementation, w is sparse if phi_vector is sparse.
%
%  options:    (an optional) structure with some of the following fields to
%              customize the behavior of the optimization algorithm:
%
%   lambda      The regularization constant (default: 1/n).
%
%   gap_threshold **STOPPING CRITERION**
%               Stop the algorithm once the duality gap falls below this
%               threshold. Note that the default of 0.1 is equivalent
%               to the criterion used in svm_struct_learn Matlab wrapper.
%               (default: 0.1).
%
%   num_passes  Maximum number of passes through the data before the
%               algorithm stops (default: 200)
%
%   debug       Boolean flag whether to track the primal objective, dual
%               objective, and training error (makes the code about 3x
%               slower given the extra two passes through data).
%               (default: 0)
%   do_linesearch
%               Boolean flag whether to use line-search. (default: 1)
%   time_budget Number of minutes after which the algorithm should terminate.
%               Useful if the solver is run on a cluster with some runtime
%               limits. (default: inf)
%   test_data   Struct with two fields: patterns and labels, which are cell
%               arrays of the same form as the training data. If provided the
%               logging will also evaluate the test error.
%               (default: [])
%
% Outputs:
%   model       model.w contains the parameters;
%               model.ell contains b'*alpha which is useful to compute
%               duality gap, etc.
%   progress    Primal objective, duality gap etc as the algorithm progresses,
%               can be used to visualize the convergence.
%
% Authors: Gauthier Gidel, Tony Jebara, Simon Lacoste-Julien
% Web: https://github.com/GauthierGidel/SP-FW
%
% Relevant Publication:
%       G. Gidel, T. Jebara, S. Lacoste-Julien
%       Frank-Wolfe Algorithms for Saddle Point problems,
%       arXiv:1610.07797, 2016


% == getting the problem description:
phi = param.featureFn; % for \phi(x,y) feature mapping
loss = param.lossFn; % for L(ytruth, ypred) loss function
if isfield(param, 'constraintFn')
    % for backward compatibility with svm-struct-learn
    maxOracle = param.constraintFn;
else
    maxOracle = param.oracleFn; % loss-augmented decoding function
end

patterns = param.patterns; % {x_i} cell array
labels = param.labels; % {y_i} cell array
n = length(patterns); % number of training examples

% == parse the options
options_default = defaultOptions(n);
if (nargin >= 2)
    options = processOptions(options, options_default);
else
    options = options_default;
end

% general initializations
lambda = options.lambda;
beta = options.beta;
phi1 = phi(param, patterns{1}, labels{1}); % use first example to determine dimension
d = length(phi1) % dimension of feature mapping
using_sparse_features = issparse(phi1);
progress = [];
if options.solution
    w_star = options.w_star;
end
% === Initialization ===
% set w to zero vector
% (corresponds to setting all the mass of each dual variable block \alpha_(i)
% on the true label y_i coordinate [i.e. \alpha_i(y) =
% Kronecker-delta(y,y_i)] using notation from Appendix E of paper).

% w: d x 1: store the current parameter iterate
% wMat: d x n matrix, wMat(:,i) is storing w_i (in Alg. 4 notation) for example i.
%    Using implicit dual variable notation, we would have w_i = A \alpha_[i]
%    -- see section 5, "application to the Structural SVM"
if using_sparse_features
    model.w = sparse(d,1);
else
    model.w = zeros(d,1);
end

% logging
progress.primal = [];
progress.dual = [];
progress.gap = [];
progress.eff_pass = [];
progress.train_error = [];
if (isstruct(options.test_data) && isfield(options.test_data, 'patterns'))
    progress.test_error = [];
end
L=0; %approx of the norm of the gradient
for i = 1:n
    % solve the loss-augmented inference for point i
    model.w = zeros(d,1);
    ystar_i = maxOracle(param, model, patterns{i}, labels{i});
    % define the update quantities:
    % [note that lambda*w_s is subgradient of 1/n*H_i(w) ]
    % psi_i(y) := phi(x_i,y_i) - phi(x_i, y)
    psi_i =   phi(param, patterns{i}, labels{i}) ...
            - phi(param, patterns{i}, ystar_i);
    loss_i = loss(param, labels{i}, ystar_i);
    L = L + 1/n*norm(psi_i);
    % sanity check, if this assertion fails, probably there is a bug in the
end
if options.solution == 1
    prim = 0;
    f_star = 0;
    f= 0;
    for i = 1:n
        % solve the loss-augmented inference for point i
        model.w = w_star;
        ystar_i_opt = maxOracle(param, model, patterns{i}, labels{i});
        % define the update quantities:
        % [note that lambda*w_s is subgradient of 1/n*H_i(w) ]
        % psi_i(y) := phi(x_i,y_i) - phi(x_i, y)
        psi_i_opt =   phi(param, patterns{i}, labels{i}) ...
                - phi(param, patterns{i}, ystar_i_opt);
        loss_i_opt = loss(param, labels{i}, ystar_i_opt);
        f_star = f_star +  1/n*( loss_i_opt - psi_i_opt'*w_star);
        % sanity check, if this assertion fails, probably there is a bug in the
    end
    % prim = f - f_star;
    % progress.primal = [prim];
    % progress.eff_pass = [0];
end
model.w = zeros(d,1)
fprintf('running Projected subgradient on %d examples. The options are as follows:\n', length(patterns));
options

tic();

k = 0;
% === Main loop ====
for p=1:options.num_passes

    for dummy = 1:n
        if using_sparse_features
            g_w = sparse(d,1);
            w_y_s = sparse(d,1);
        else
            g_w = zeros(d,1);
            w_y_s = zeros(d,1);
        end
        ell_s = 0;
        f = 0;
        i = randi(n);

        % solve the loss-augmented inference for point i
        ystar_i = maxOracle(param, model, patterns{i}, labels{i});

        % define the update quantities:
        % [note that lambda*w_s is subgradient of 1/n*H_i(w) ]
        % psi_i(y) := phi(x_i,y_i) - phi(x_i, y)
        %%%compute the subgradient%%%
        psi_i =   phi(param, patterns{i}, labels{i}) ...
                - phi(param, patterns{i}, ystar_i);
        g_w =  g_w - 1/n * psi_i; - lambda* model.w;
        % sanity check, if this assertion fails, probably there is a bug in the
        % maxOracle or in the featuremap

        %w_y = Ay_s
        % t he true w_s only contains the opposite of the biggest (in absolute value) coordinate times beta
        % w_y = model.w_y;
        % [~,i] = max(abs(w_y-lambda*model.w));
        % w_s(i) = beta*sign(w_y(i));
        % assert(sum(abs(w_s))<=beta)
        % % compute duality gap:
        % gap_w = -(model.w-w_s)'*(w_y-lambda*model.w);
        % gap_y = ell_s -  model.ell -  model.w'* (w_y_s - model.w_y);
        % gap = gap_w + gap_y;    % keyboard
        % % get the step-size gamma:
        %%%% !!!! No line search for saddle points
        % we use the fixed step-size schedule
        gap = -(model.w- beta * sign(g_w))' * g_w;
        % assert(gap>0)
        h = param.stepsize./(L.*sqrt(1+k./n));
        % We use the adaptative and asymetric step size
        % C = 2*beta*(lambda + 1./lambda);
        % gamma_w = min(gap_w./(2*C),1);
        % gamma_y = min(gap_y./(20*C),1) ;
        % % gamma_y = gamma;
        % gamma_w = gamma;

        % % stop if duality gap is below threshold:
        % if gap <= options.gap_threshold
        %     fprintf('Duality gap below threshold -- stopping!\n')
        %     fprintf('current gap: %g, gap_threshold: %g\n', gap, options.gap_threshold)
        %     fprintf('Reached at iteration %d.\n', k)
        %     break % exit loop!
        % elseif  (options.debug && mod(k,options.debug_iter)==0)
        %     fprintf('Duality gap check: gap = %g at iteration %d\n', gap, k);
        % end
        % finally update the weights and ell variables
        %check If we do a away step

        %
        % model.w_y = (1-gamma)*model.w_y   + gamma*w_y_s;
        model.w   = proj_l1_ball(model.w   - h*g_w,beta);
        % model.ell = (1-gamma)*model.ell + gamma*ell_s;
        % debug: compute objective and duality gap. do not use this flag for
        % timing the optimization, since it is very costly!
        k = k+1;
    end
    if options.solution
        f=0;
        g_w= 0;
        for i = 1:n

            % solve the loss-augmented inference for point i
            ystar_i = maxOracle(param, model, patterns{i}, labels{i});

            % define the update quantities:
            % [note that lambda*w_s is subgradient of 1/n*H_i(w) ]
            % psi_i(y) := phi(x_i,y_i) - phi(x_i, y)
            %%%compute the subgradient%%%
            psi_i =   phi(param, patterns{i}, labels{i}) ...
                    - phi(param, patterns{i}, ystar_i);
            loss_i = loss(param, labels{i}, ystar_i);
            g_w =  g_w - 1/n * psi_i;
            f = f + 1/n *(loss_i- psi_i'*model.w );
            % sanity check, if this assertion fails, probably there is a bug in the
            % maxOracle or in the featuremap

        end
        gap = -(model.w- beta * sign(g_w))' * g_w;
        primal = f - f_star;
        progress.primal = [progress.primal; primal];
    end
    if (options.debug&& mod(k,n)==0)
        progress.gap = [progress.gap; gap];
        %%compute L(w^k,y^*) - L(w^*,y^k) =
        % primal = -objective_function(model.w, model.w_y,model.ell);
        % % gap = duality_gap(param, maxOracle, model, lambda);
        % % primal = f+gap; % a cheaper alternative to get the primal value
        % train_error = average_loss(param, maxOracle, model);
        % fprintf('pass %d (iteration %d), SVM primal = %f, duality gap = %f, train_error = %f \n', ...
        %     p, k+1, primal, gap, train_error);

        % progress.primal = [progress.primal; primal];
        % % progress.dual = [progress.dual; f];
        progress.eff_pass = [progress.eff_pass; k/n];
        % progress.train_error = [progress.train_error; train_error];
        % if (isstruct(options.test_data) && isfield(options.test_data, 'patterns'))
        %     param_debug = param;
        %     param_debug.patterns = options.test_data.patterns;
        %     param_debug.labels = options.test_data.labels;
        %     test_error = average_loss(param_debug, maxOracle, model);
        %     progress.test_error = [progress.test_error; test_error];
        % end
end

    % % time-budget exceeded?
    % t_elapsed = toc();
    % if (t_elapsed/60 > options.time_budget)
    %     fprintf('time budget exceeded.\n');
    %     return
    % end
end

end %Function

function options = defaultOptions(n)

options = [];
options.num_passes = 200;
options.do_line_search = 1;
options.time_budget = inf;
options.debug = 0;
options.lambda = 1/n;
options.test_data = [];
options.gap_threshold = 0.1;

end % defaultOptions
