% Applies the structured SVM to the OCR dataset by Ben Taskar. The structured
% model considered here is the standard chain graph, with the pixel values of
% the digit as unary features and a transition matrix of size num_states^2 as
% a pairwise potential. Additionally, we include a unary bias term for the first
% and last symbol in the sequence.

addpath(genpath('../../solvers/'));
addpath('helpers');


data_name = 'ocr';
[patterns_train, labels_train, patterns_test, labels_test] = loadOCRData(data_name, '../../data/');

% create problem structure:
subset_size = 100;
param = [];
param.patterns = patterns_train(1:subset_size);
param.labels = labels_train(1:subset_size);
param.lossFn = @chain_loss;
param.oracleFn = @chain_oracle;
param.featureFn = @chain_featuremap;

% options structure:
options = [];
options.alpha = 2;
options.lambda = 0.0;
options.gap_threshold = 0.00001; % duality gap stopping criterion
options.num_passes = 20000 ; % max number of passes through data
options.do_line_search = 0;
options.sample = 'uniform';
options.debug_iter = 100;
beta = 5;

% Compute the approximal optimal value
options.solution = 0;
options.debug = 1;
options.average = 0;
options.beta = beta;

param.stepsize =1;
[model, progress] = solverSP_BCFW(param, options);
w_star = model.w;
options.solution = 1;

options.w_star = w_star;


options.num_passes = 1000 ; % max number of passes through data

options.beta = beta;
[model1,progress1] = solverSubgradient(param,options);
[model2,progress2] = solverStoSubgradient(param,options);
param.stepsize=0.1;
[model3,progress3] = solverStoSubgradient(param,options);
[model4,progress4] = solverSP_FW(param,options);
options.alpha = 1;
[model5,progress5] = solverSP_FW(param,options);
options.average =1;
[model6,progress6] = solverSP_BCFW(param,options);
loglog(progress1.eff_pass, progress1.primal,'b');
hold on
loglog(progress2.eff_pass, progress2.primal,'c')
loglog(progress3.eff_pass, progress3.primal,'g');
loglog(progress4.eff_pass, progress4.primal,'r');
loglog(progress5.eff_pass, progress5.primal,'c');
loglog(progress6.eff_pass, progress6.primal,'y');
legend('Subgradient','SSG','SSG2','SPFW','SPFW2','BCSWFW')
hold off
