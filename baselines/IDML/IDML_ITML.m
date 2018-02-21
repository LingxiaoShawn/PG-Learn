function result = IDML_ITML(data, params_constant, params_alterable)
% Run IDML graph learning algorithm with ITML as metric learning part. 
% Algorithm is proposed in Paramveer Dhillon's Inference Driven Metric 
% Learning (IDML) for Graph Construction.
% =======================================
% Input:
%   data: processed data, divided into labeled data and unlabeled data.
%   | data.XTrain - n*d feature matrix for training data
%   | data.yTrain - n*1 class labels for training data (label starts from 0)
%   | data.XValid - n*d feature matrix for validation data
%   | data.yValid - n*1 class labels for validation data (label starts from 0)
%   | data.XTest - n*d feature matrix for test data
%   | data.yTest - n*1 class labels for test data (label starts from 0) 
%   -------------------------------------------------------------------
%   params_constant:
%   | params_constant.graph_type - 1=euclidean (current only support 1)
%   | params_constant.isSparse - use sparse operation or not (suggest sparse)
%   | params_constant.tic_started - whether started a timer
%   -------------------------------------------------------------------
%   params_alterable:
%   | params_alterable.time - given time for running 
%   | params_alterable.k - k for KNN 
%   | (optional)params_alterable.param1 - denotes k, used for RanSet Parallel 
%   | (optional)params_alterable.param2 - denotes rho, used for Ranset Parallel
%   | (optional)params_alterable.param3 - denotes beta, used for Ranset Parallel
%   -------------------------------------------------------------------
% Output:
%   result
%   | result.val_acc - achieved validation accuracy 
%   | result.test_acc - corresponding test accuracy
%   | result.timing - a row array of all corresponding time history 
%   | result.k - corresponding k 
%   | result.rho - corresponding rho  
%   | result.beta - corresponding beta
%   | result.ori_test_acc - history of original test accuracy
%   | result.ori_val_acc - history of original validation accuracy
%   -------------------------------------------------------------------
%% parameters
if ~isfield(params_constant, 'tic_started')
    tic;
else
    if ~params_constant.tic_started
        tic;
    end
end
graph_type = params_constant.graph_type;
isSparse = params_constant.isSparse;
time_limit = params_alterable.time;
alpha = 0.99;
n_iter = 100;
%% check received alterable params
if isfield(params_alterable, 'param1') % k
    k = params_alterable.param1;
else
    k = params_alterable.k;
end

if isfield(params_alterable, 'param2') % rho
    rho = params_alterable.param2;
else
    rho = 1;
end

if isfield(params_alterable, 'param3') % beta
    beta = params_alterable.param3;
else
    beta = 0.05;
end
%% load data
XTrain = data.XTrain;
yTrain = data.yTrain;
XValid = data.XValid;
yValid = data.yValid;
XTest = data.XTest;
yTest = data.yTest;
y = [yTrain;yValid;yTest];
X = [XTrain; XValid; XTest];
n = size(X,1);
nTrain = length(yTrain); % number of training samples
nValid = length(yValid); % number of validation samples
nTest = length(yTest); % number of unlabeled samples
nTV = nTrain + nValid; % number of labeled samples
ft = ind2vec(yTrain'+1)'; % one-hot encode for training label
fl = ind2vec([yTrain;yValid]'+1)'; % one-hot encode for train+validation label
ft = [ft; sparse(n-nTrain, size(ft,2))];
fl = [fl; sparse(n-nTV, size(ft,2))];

ori_val_acc = [];
ori_test_acc = [];
timing = [];
train_index = 1:nTrain;
unlabeled_index = (1:nTest) + nTV;
%% get all parameters of ITML 
A0 = eye(size(X,2)); % default setting
sigma0 = mean(pdist(X));
sigma = rho*sigma0;
% determine similarity/dissimilarity constraints from the true labels
[l, u] = ComputeDistanceExtremes(X, 5, 95, A0); % default setting
% construct C matrix(constraints) of ITML algorithm based on training data
%     C: 4 column matrix
%        column 1, 2: index of constrained points.  Indexes between 1 and n
%        column 3: 1 if points are similar, -1 if dissimilar
%        column 4: right-hand side (lower or upper bound, depending on 
%                  whether points are similar or dissimilar)
C = zeros(nTrain * (nTrain - 1) / 2, 4);
kk = 1;
for i=1:nTrain-1
    for j=i+1:nTrain
        if yTrain(i) == yTrain(j)
            C(kk,:) = [i j 1 l];
        else
            C(kk,:) = [i j -1 u];
        end
        kk = kk + 1;
    end
end

% params: algorithm parameters - see see SetDefaultParams for defaults
%           params.thresh: algorithm convergence threshold
%           params.gamma: gamma value for slack variables
%           params.max_iters: maximum number of iterations
params = struct();
params = SetDefaultParams(params);
params.max_iters = 100000;
%% repeat IDML loop
while toc < time_limit
%========= repeat =============% 

% ITML(supervised metric learning)
fprintf('Start ITML!\n');
A = ItmlAlg(C, X, A0, params);

% construct knn graph
fprintf('Start construct knn graph!\n');
W = SimGaussianGraph_KNN(X, k, A, graph_type, isSparse);
W = W.^(1/(2*sigma^2));

% label propagation 
fprintf('Start label propagation!\n');
if params_constant.split_val 
    F = LabelPropagation(W, ft, alpha, n_iter);
else
    F = LabelPropagation(W, fl, alpha, n_iter);
end
% normalize F 
F = bsxfun(@rdivide, F, sum(F)); % CMN
F = bsxfun(@rdivide, F, sum(F,2)); 

% get validation accuracy 
fv =  (vec2ind(F(nTrain+1:nTV,:)') - 1)';
val_acc = sum(fv == yValid) / length(yValid)
ori_val_acc = [ori_val_acc, val_acc];

% get test accuracy
if params_constant.split_val 
    F_test = LabelPropagation(W, fl, alpha, n_iter);
    % normalize F 
    F_test = bsxfun(@rdivide, F_test, sum(F_test)); % CMN
    F_test = bsxfun(@rdivide, F_test, sum(F_test,2)); 
else
    F_test = F;
end

ftest = (vec2ind(F_test(nTV+1:end,:)') - 1)'; 
test_acc = sum(ftest == yTest) / nTest
ori_test_acc = [ori_test_acc, test_acc];
timing = [timing, toc];


% select low entropy instances that below entropy beta = 0.05
F_unlabeled = F(nTV+1:end,:);
entropy_unlabeled = full(sum(-F_unlabeled.*log(F_unlabeled),2));
fprintf('Minimum entropy: %d\n', min(entropy_unlabeled));
chosen_index = unlabeled_index(entropy_unlabeled < min(entropy_unlabeled) + beta);

% remove old-added elements
chosen_index = setdiff(chosen_index, yTrain');
% check 
if isempty(chosen_index)
    fprintf('No added label instance!\n');
else
    fprintf('Number of new labeled data: %d\n', length(chosen_index)); 
end
predicted_label = vec2ind(F(chosen_index,:)') - 1

% update C
for ii = 1:length(chosen_index)
    i = chosen_index(ii);
    y_i = predicted_label(ii);
    C_add = zeros(length(train_index),4);
    C_add(:,1) = i;
    same_index = train_index(yTrain == y_i);
    diff_index = train_index(yTrain ~= y_i);
    n_same = length(same_index);
    C_add(1:n_same,2) = same_index;
    C_add(1:n_same, [3 4]) = repmat([1 l], n_same, 1);
    C_add(n_same+1:end, 2) = diff_index;
    C_add(n_same+1:end, [3 4]) = repmat([-1 u], length(diff_index), 1);
    C = [C;C_add];
    train_index = [train_index, i];
    yTrain = [yTrain; y_i];
    % update ft and fl
    ft(i, y_i + 1) = 1;
    fl(i, y_i + 1) = 1;
    if y_i == y(i)
        fprintf('added RIGHT label instance!\n');
    else
        fprintf('added WRONG label instance!\n');
    end   
end
A0 = A;

% convergence condition
if length(train_index) >= 0.8*n
    break;
end

end
%% return result
result.ori_test_acc = ori_test_acc;
result.ori_val_acc = ori_val_acc;
if params_constant.split_val 
    [result.val_acc, index]= max(ori_val_acc);
    result.test_acc = ori_test_acc(index);
else
    result.val_acc = 0; % didn't use validation set
    result.test_acc = ori_test_acc(end);
end
result.timing = timing;
result.k = k;
result.rho = rho;
result.beta = beta;
end
