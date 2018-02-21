function result = PGLearn(data, params_constant, params_alterable)
% Run PG-Learn algorithm. Single thread version. 
% ==============================================
% Input:
%   data: the input data, XValid and yValid can be empty array []
%   | data.XTrain - n*d feature matrix for training data
%   | data.yTrain - n*1 class labels for training data (label starts from 0)
%   | data.XValid - n*d feature matrix for validation data
%   | data.yValid - n*1 class labels for validation data (label starts from 0)
%   | data.XTest - n*d feature matrix for test data
%   | data.yTest - n*1 class labels for test data (label starts from 0)
%   ------------------------------------------------------------------
%   params_constant: parameters that stay costant for parallel settting
%   | params_constant.lambda - learning rate for gradient-based method
%   | params_constant.batch_size - batch size for learning vector a
%   | params_constant.graph_type - 1=euclidean (current only support 1)
%   | params_constant.isSparse - use sparse operation or not (suggest sparse)
%   | params_constant.tic_started - whether started a timer
%   | params_constant.method - PG-Learn or MinEnt 
%   |         (the MinEnt is not exactly same  with Zhu's proposal, we only
%   |          use the loss function of MinEnt)
%   ------------------------------------------------------------------
%   params_alterable: parameters that is changeable for parallel settting
%   | params_alterable.time - given time for running
%   | params_alterable.k or param1 - k for KNN
%   | params_alterable.init_point or param2 - init a 
%   ------------------------------------------------
% Output:
%   result: 
%   | result.val_acc - achieved validation accuracy 
%   | result.test_acc - achieved test accuracy
%   | result.init_point - learned a 
%   | result.k - the final k corresponding to val_acc
%   | result.max_v_history - a row array of validation accuracy history 
%   | result.max_t_history - a row array of test accuracy history 
%   | result.max_g_history - a row array of loss value history
%   | result.ori_val_acc - a row array of original validation accuracy history 
%   | result.ori_test_acc - a row array of original test accuracy history
%   | result.timing - a row array of all corresponding time history
%   | result.g - a row array of all corresponding loss value history
%   ------------------------------------------------------------------
%% load data
XTrain = data.XTrain;
yTrain = data.yTrain;
XValid = data.XValid;
yValid = data.yValid;
XTest = data.XTest;
yTest = data.yTest;
X = [XTrain; XValid; XTest];
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
lambda = params_constant.lambda;
batch_size = params_constant.batch_size;
graph_type = params_constant.graph_type;
isSparse = params_constant.isSparse;

time_limit = params_alterable.time;
if isfield(params_alterable, 'param1') % k
    k = params_alterable.param1;
else
    k = params_alterable.k;
end
if isfield(params_alterable, 'param2') % rho
    sigma_0 = mean(pdist(X));
    rho = params_alterable.param2;
    a = 1./(ones(1,size(X,2)) * (rho*sigma_0)^2);  
else
    a = params_alterable.init_point;
end
%% configuration
precision = 1e-4;
n_converged_limit = 5;
n_converged = 0;
max_iter = 50; % number of iteration for matrix inversion  
alpha = 0.99; % teleport probability
dim = size(X, 2); % feature dimension 
r = zeros(batch_size,1); % random coordinates of a
old_a_r = zeros(batch_size,1); % old part of a 
%% save histories 
ori_test_acc = [];
ori_val_acc = [];
max_v_history = [];
max_t_history = [];
max_g_history = [];
timing = [];
max_a = 0;
max_v = 0; 
max_t = 0;
max_g = 0;
test_acc = 0;
%% local search
result.converged = false;
% set starting point
rng('shuffle');
mini_epoch = 1;
g = []; % record loss value
old_g = 0;
while toc <= time_limit
    fprintf('===============start loop=============\n');
    t1 = toc;
    % construct or update graph
    if mini_epoch == 1
        [W, W_PreKNNExp] = SimGaussianGraph_KNN(X, k, a, graph_type, isSparse);
    else
        [W, W_PreKNNExp] = SimGaussianGraph_KNN(X, k, a, graph_type, isSparse, W_PreKNNExp, r, old_a_r);
    end
    % label propagation 
    [F, P] = LabelPropagation(W, ft, alpha, max_iter);
    % predict label as the label with maximum probability
    fv =  (vec2ind(F(nTrain+1:nTrain+nValid,:)') - 1)';
    val_acc = sum(fv == yValid) / nValid
    % if achieve higher validation accuracy, record test accuracy
    if val_acc > max_v
        max_v = val_acc;
        % calculate test_acc, using all labeled data
        F_test = LabelPropagation(W, fl, alpha, max_iter);
        ftest = (vec2ind(F_test(nTV+1:end,:)') - 1)'; 
        test_acc = sum(ftest == yTest) / nTest 
        max_t = test_acc;
        max_a = a;   
        max_g = -1;
    end
    % choose coordinates
    r = randsample(dim, batch_size);
    % compute partialW with respect to a
    [partialW, S] = PartialW(X(:,r), W, graph_type);
    % compute partialP and partial F
    W_inv = spfun(@(x) 1./x, W);
    if ~isSparse
        P_divide_W  = P.*full(W_inv);
        partialP = bsxfun(@times, partialW, P_divide_W) - 1/2*...
                   bsxfun(@times, P.*(P_divide_W.^2), S + permute(S,[2 1 3])); % n*n*d full matrix
        partialF = alpha * mtimesx(partialP, F);
    else
        P_divide_W = P.*W_inv;
        partialP = partialW(:,:) .* repmat(P_divide_W, 1, batch_size) - ...
                   1/2 * repmat(P.*(P_divide_W.^2), 1, batch_size).*...
                   reshape((S+permute(S, [2 1 3])), size(F,1), []);  % n*nd 2d sparse matrix
        partialF = full(permute(reshape(alpha*F'*partialP, ...
                        size(F,2),size(F,1),[]), [2 1 3])); % n*n*d 3d full matrix
    end
    % compute partial F iteratively
    base = partialF;
    transit = full(alpha * P);
    for iter = 1:max_iter
        partialF = mtimesx(transit, partialF) + base;
    end  
    % normalize F and partial F
    F = full(F);
    sum2_F = sum(F,2); % is there any zero?
    sum2_partialF = sum(partialF, 2); % size: n*1*batch_size
    F = bsxfun(@rdivide, F, sum2_F);
    % partialF_new = (partialF - F_new.* row_sum_partialF)./row_sum_F
    partialF = bsxfun(@rdivide, partialF - bsxfun(@times, F, sum2_partialF), sum2_F); 
    
    if strcmp(params_constant.method, 'PG-Learn')
        % compute partial g and g
        Fl = F(1:nTV,:);
        partialFl = partialF(1:nTV,:,:);
        [new_g, partialG] = LossPGLearn(Fl, partialFl, [yTrain;yValid]);        
    end
    if strcmp(params_constant.method, 'MinEnt')
        % compute partial g and g
        Fu = F(nTV+1:end,:);
        partialFu = partialF(nTV+1:end,:,:);
        [new_g, partialG] = LossMinEnt(Fu, partialFu); 
    end   
    % update a
    old_a_r = a(r);
    a(r) = max(0, a(r) - lambda * partialG); % batch coordinate update
    % save histories
    g = [g, new_g];
    if max_g == -1
        max_g = new_g;
    end
    max_t_history = [max_t_history,max_t];  
    max_v_history = [max_v_history,max_v];
    max_g_history = [max_g_history,max_g];
    ori_val_acc = [ori_val_acc, val_acc];
    ori_test_acc = [ori_test_acc, test_acc];
    timing = [timing, toc];
    mini_epoch = mini_epoch + 1;
    fprintf('time of one update: %ds\n',toc-t1); 
    % adaptive change learning rate
    if new_g > 1.005*old_g 
        lambda = lambda * 0.9;       
    end
    % convergence condition
    if val_acc == 1
        % prevent overfit
        result.converged = true;
        break;  
    else
        if abs(new_g - old_g) < precision
            n_converged = n_converged + 1;
            if n_converged >= n_converged_limit 
                result.converged = true;
                break;
            end
        else
            n_converged = 0;
        end
    end
    test_acc
    old_g = new_g
end
%% return result
result.val_acc = max_v;
result.test_acc = max_t;
result.init_point = max_a;
result.max_v_history = max_v_history;
result.max_t_history = max_t_history;
result.max_g_history = max_g_history;
result.ori_val_acc = ori_val_acc;
result.ori_test_acc = ori_test_acc;
result.timing = timing;
result.g = g;
result.k = k;
result.time_cost = toc;
