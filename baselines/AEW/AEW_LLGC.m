function result = AEW_LLGC(data, params_constant, params_alterable)
% Run AEW algorithm with LLGC label propagation. 
% AEW is proposed in Masayuki Karasuyama's Adaptive 
% edge weighting for graph-based learning algorithms. 
% ===================================================
% Input:
%   data: processed data, divided into labeled data and unlabeled data.
%   | data.XTrain - n*d feature matrix for training data
%   | data.yTrain - n*1 class labels for training data (label starts from 0)
%   | data.XValid - n*d feature matrix for validation data
%   | data.yValid - n*1 class labels for validation data (label starts from 0)
%   | data.XTest - n*d feature matrix for test data
%   | data.yTest - n*1 class labels for test data (label starts from 0) 
%   -------------------------------------------------------------------
%   params_constant: this argument isn't used by AEW_LLGC, which is only 
%           used to keep the input having same form with other algorithms.
%   params_alterable:
%   | params_alterable.k or param1 - k for KNN
%   -------------------------------------------------------------------
% Output:
%   result:
%   | result.val_acc - achieved validation accuracy 
%   | result.test_acc - achieved test accuracy
%   | result.timing - a row array of all corresponding time history
%   | result.k - choosed k
%   ------------------------------------------------------------------- 
%% parameters
if isfield(params_alterable, 'param1') % k
    k = params_alterable.param1;
else
    k = params_alterable.k;
end
alpha = 0.99;
max_iter = 50;
%% load data
XTrain = data.XTrain;
yTrain = data.yTrain;
XValid = data.XValid;
yValid = data.yValid;
XTest = data.XTest;
yTest = data.yTest;
X = [XTrain; XValid; XTest];
n = size(X,1);
nTrain = length(yTrain); % number of training samples
nValid = length(yValid); % number of validation samples
nTest = length(yTest); % number of unlabeled samples
nTV = nTrain + nValid; % number of labeled samples
ft = ind2vec(yTrain'+1)'; % one-hot encode for training label
fl = ind2vec([yTrain;yValid]'+1)'; % one-hot encode for train+validation label
%% AEW
param.k = k; % The number of neighbors in kNN graph
param.sigma = 'local-scaling'; % Kernel parameter heuristics 'median' or 'local-scaling'
param.max_iter = 100; % default
%---------- start ----------%
fprintf('Optimizing edge weights by AEW\n');
[W W0] = AEW(X',param);
fprintf('Estimating labels by LLGC... ');
F_val = LabelPropagation(W, ft, alpha, max_iter);
F_test = LabelPropagation(W, fl, alpha, max_iter);
fprintf('done\n');
% predict label as the label with maximum probability
fVal = (vec2ind(F_val(nTrain+1:nTV,:)') - 1)';
fTest = (vec2ind(F_test(nTV+1:end,:)') - 1)';
val_acc = sum(fVal == yValid) / length(yValid);
test_acc = sum(fTest == yTest) / (n-nTV);
%% return result
result.val_acc = val_acc;
result.test_acc = test_acc;
result.timing = [toc];
result.k = k;
