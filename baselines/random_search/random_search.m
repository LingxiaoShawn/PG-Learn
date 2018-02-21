function result = random_search(data, params_constant, params_alterable)
% Run random search in given time limitation
% ==========================================
% Inputs:
%   data: processed data, divided into labeled data and unlabeled data.
%   | data.XTrain - n*d feature matrix for training data
%   | data.yTrain - n*1 class labels for training data (label starts from 0)
%   | data.XValid - n*d feature matrix for validation data
%   | data.yValid - n*1 class labels for validation data (label starts from 0)
%   | data.XTest - n*d feature matrix for test data
%   | data.yTest - n*1 class labels for test data (label starts from 0) 
%   -------------------------------------------------------------------
%   params_constant: parameters that stay costant for parallel settting
%   | params_constant.k_range - searching range for k of KNN
%   | params_constant.graph_type - 1=euclidean (current only support 1)
%   | params_constant.isSparse - use sparse operation or not (suggest sparse)
%   | params_constant.tic_started - whether started a timer
%   -------------------------------------------------------------------
%   params_alterable: parameters that is changeable for parallel settting
%   | params_alterable.time - given time for running
%   | params_alterable.a_range - searching range for each element of a 
%   -------------------------------------------------------------------  
% Outputs:
%   result:
%   | result.val_acc - achieved validation accuracy 
%   | result.test_acc - achieved test accuracy
%   | result.init_point - learned vector a
%   | result.max_v_history - a row array of validation accuracy history 
%   | result.max_t_history - a row array of test accuracy history 
%   | result.timing - a row array of all corresponding time history
%   | result.k - choosed k
%   -------------------------------------------------------------------
%% load data
XTrain = data.XTrain;
yTrain = data.yTrain;
XValid = data.XValid;
yValid = data.yValid;
XTest = data.XTest;
yTest = data.yTest;
X = [XTrain; XValid; XTest];
dim = size(X, 2);
nTrain = length(yTrain); % number of training samples
nValid = length(yValid); % number of validation samples
nTest = length(yTest); % number of unlabeled samples
nTV = nTrain + nValid; % number of labeled samples
ft = ind2vec(yTrain'+1)'; % one-hot encode for training label
fl = ind2vec([yTrain;yValid]'+1)'; % one-hot encode for train+validation label
%% parameters
% whether to start a new timer
if ~isfield(params_constant, 'tic_started')
    tic;
else
    if ~params_constant.tic_started
        tic;
    end
end
k_range = params_constant.k_range;
graph_type = params_constant.graph_type;
isSparse = params_constant.isSparse;
time_limit = params_alterable.time;
a_range = params_alterable.a_range;
n_iter = 50; % number of iteration for matrix inversion  
alpha = 0.99; % teleport probability
%% random search
max_v = 0; 
max_t = 0;
max_v_history = [];
max_t_history = []; 
timing = [];

while toc <= time_limit
    rng('shuffle');
    a = 1./(rand(1,dim) * (a_range(2) - a_range(1)) + a_range(1));    
    k = randi(k_range);
    %W = construct_graph_func(X, k, a, sim_func); 
    W = SimGaussianGraph_KNN(X, k, a, graph_type, isSparse);
    F = LabelPropagation(W, ft, alpha, n_iter);
    % predict label as the label with maximum probability
    fv =  (vec2ind(F(nTrain+1:nTrain+nValid,:)') - 1)';
    new_val_acc = sum(fv == yValid) / nValid;
    
    if new_val_acc >= max_v
        max_v = new_val_acc;
        F_test = LabelPropagation(W, fl, alpha, n_iter);
        ftest = (vec2ind(F_test(nTV+1:end,:)') - 1)'; 
        max_t = sum(ftest == yTest)/nTest;
        best_a = a;
        best_k = k;
    end
    
    max_v_history = [max_v_history, max_v];
    max_t_history = [max_t_history, max_t];
    timing = [timing, toc];
end
%% return results
result.val_acc = max_v;
result.test_acc = max_t;
result.init_point = best_a;
result.k = best_k;
result.max_v_history = max_v_history;
result.max_t_history = max_t_history;
result.timing = timing;
