function combinedResult = RndSet_parallel_framework(algFunc, data, infoForNaming, params)
% Run algorithm in parallel setting, where each run is starting from random
% parameters of given sets. For each thread, it will restart an random run
% if the original run is converged before the given time. 
% ===================================
% Inputs
%   algFunc: the function of single-thread graph learning algorithm,
%   |            result = @func(data, params_constant, params_alterable)
%   | Interface of algFunc:
%   |   data: same as below 
%   |   params_constant: look at SetConstantParams function
%   |   params_alterable:
%   |   | params_alterable.time - required
%   |   | params_alterable.paramNSet - required (N = 1,2,...)
%   |   ----------------------------------------
%   |   result:
%   |   | result.val_acc - required
%   |   | result.test_acc - required
%   |   | result.timing - required
%   |   | (optional)result.max_v_history - only required if params.showAllHistory = true
%   |   | (optional)result.max_t_history - only required if params.showAllHistory = true
%   ------------------------------------------------------------------    
%   data: processed data, divided into labeled data and unlabeled data.
%   | data.XTrain - n*d feature matrix for training data
%   | data.yTrain - n*1 class labels for training data (label starts from 0)
%   | data.XValid - n*d feature matrix for validation data
%   | data.yValid - n*1 class labels for validation data (label starts from 0)
%   | data.XTest - n*d feature matrix for test data
%   | data.yTest - n*1 class labels for test data (label starts from 0)
%   ------------------------------------------------------------------
%   infoForNaming: a string contains all specific information.
%   params: parameters for parallel setting as well as learning setting.
%   | params.nThreads - number of threads used 
%   | params.totalHours - number of time used
%   | params.graphType - 1=euclidean 
%   | params.isSparse - use sparse operation or not
%   | params.isNewPool - whether start a new parallel pool 
%   | params.showAllHistory - whether show all history, this can only be
%                  used for algFunc that return result.max_v_history and
%                  result.max_t_history
%   | params.param1Set - (k) alterable parameter sent into algFunc
%   | params.param2Set - (rho or sigma) alterable parameter sent into algFunc
%   | params.param3Set - (beta) alterable parameter sent into algFunc
%   | params.paramNSet - (N can be any, less than n_max) alterable parameter 
%                         sent into algFunc
%   ------------------------------------------------------------------             
% Outputs
%   combinedResult: combined result for parallel setting.
%   | combinedResult.combined_max_v_history - validation accuracy
%   | combinedResult.combined_max_t_history - test accuracy
%   | combinedResult.combined_timing - corresponding time
%   ------------------------------------------------------------------    
clear combinedResult
n_max = 5;
%% check args 
if nargin < 4
    fprinf('Please give all four args!');
    return;
end
params = SetParallelParams(params);
%% parallel settings
start_new_pool = params.isNewPool;
total_hours = params.totalHours;
n_parallel = params.nThreads;
single_runtime = total_hours*3600

params_constant = struct();
if isfield(params, 'graphType')
    params_constant.graph_type = params.graphType; % 1 = euclidean
end
if isfield(params, 'isSparse')
    params_constant.isSparse = params.isSparse;
end
params_constant = SetConstantParams(params_constant);
params_constant.tic_started = true;
%% Start the pool
clust = parcluster('local');
clust.NumWorkers = n_parallel + 1;
if start_new_pool
    delete(gcp('nocreate'))
end
p = gcp('nocreate');
if isempty(p)
    parpool(clust, clust.NumWorkers);
else
    n_parallel = p.NumWorkers - 1;
end
%% random select init params from given sets and run given algorithm
% create a pool to save all results
clear results_pool
pool_lengths = ones(1, n_parallel);

% check exisitance of params
n_send_params = 0;
for i = 1:n_max
    if isfield(params, sprintf('param%dSet',i))
        n_send_params = n_send_params + 1;
    else
        break;
    end
end
% run 
spmd 
    % master
    if labindex == 1
        n_finished = 0;
        % send params to all slaves
        for j = 2:numlabs
            % sampling params from given s
            send_params = struct();
            send_params.time = single_runtime;
            for i = 1:n_send_params
                send_params.(sprintf('param%d',i)) = datasample(...
                    params.(sprintf('param%dSet',i)), 1);
            end
            labSend(send_params, j);
        end      
        % keep receive results from all slaves
        active_workers = 2:n_parallel+1;
        while n_finished < n_parallel 
            for srcWkrIdx = active_workers
                if labProbe(srcWkrIdx)
                    result = labReceive(srcWkrIdx);
                    % save result
                    results_pool(srcWkrIdx-1, pool_lengths(srcWkrIdx-1)) = result;
                    pool_lengths(srcWkrIdx-1) = pool_lengths(srcWkrIdx-1) + 1;
                    if result.finished
                        n_finished = n_finished + 1;
                        active_workers = setdiff(active_workers, srcWkrIdx);
                    else
                        % sampling again and resend to slaves
                        send_params.time = single_runtime;
                        for i = 1:n_send_params
                            send_params.(sprintf('param%d',i)) = datasample(...
                                params.(sprintf('param%dSet',i)), 1);
                        end
                        labSend(send_params, srcWkrIdx);
                    end
                end
            end
        end
        fprintf('Master finished normally!\n');
    % slaves    
    else
        tic;
        while toc < single_runtime
            result = algFunc(data, params_constant, labReceive(1));
            result.finished = false;
            if toc >= single_runtime - 5
                result.finished = true;
            end
            labSend(result, 1);
            if result.finished
                while toc < single_runtime
                end
            end
        end
        fprintf('Finished in %d seconds!\n',toc);
    end
end
%% save results
clear saved_results
results_pool = results_pool{1};
pool_lengths = pool_lengths{1};
index = 1;
for i = 1:n_parallel
    for j = 1:pool_lengths(i)-1
        saved_results(index) = results_pool(i,j);
        index = index + 1;
    end
end
name = strcat(infoForNaming, sprintf('_parallel_%dtasks_%gh',...
                                          n_parallel, total_hours));
if params.showAllHistory
    lens = zeros(1, n_parallel);
    max_len = 0;
    % get all history of max_v and max_t for each thread
    clear temp
    for i = 1:n_parallel
        cur_max_v = 0;
        cur_max_t = 0;
        max_v_history = [];
        max_t_history = [];
        timing = [];
        for j = 1:pool_lengths(i)-1  
            temp1 = results_pool(i,j).max_v_history;
            temp2 = results_pool(i,j).max_t_history;
            temp2(temp1<cur_max_v) = cur_max_t;
            temp1(temp1<cur_max_v) = cur_max_v;
            max_v_history = [max_v_history, temp1];
            max_t_history = [max_t_history, temp2];
            timing = [timing, results_pool(i,j).timing];
            cur_max_v = max(cur_max_v, results_pool(i,j).val_acc);
            if cur_max_v == results_pool(i,j).val_acc
                cur_max_t = results_pool(i,j).test_acc;
            end
        end
        temp(i).max_v_history = max_v_history;
        temp(i).max_t_history = max_t_history;
        temp(i).timing = timing;
        lens(i) = length(max_v_history);
        if max_len < lens(i)
            max_len = lens(i);
        end
    end
    max_v_histories = zeros(n_parallel, max_len);
    max_t_histories = zeros(n_parallel, max_len);
    timings = zeros(n_parallel, max_len);
    for i = 1:n_parallel
        max_v_histories(i, 1:lens(i)) = temp(i).max_v_history;
        max_t_histories(i, 1:lens(i)) = temp(i).max_t_history;
        timings(i,1:lens(i)) = temp(i).timing;
    end
    % get combined history
    [combined_max_v_history, index] = max(max_v_histories,[],1);
    index = index + (0:max_len-1)*n_parallel;
    combined_max_t_history = max_t_histories(index);
    combined_timing = timings(index);
    
else
    combined_max_v_history = 1:length(saved_results);
    combined_max_t_history = 1:length(saved_results);
    combined_timing = 1:length(saved_results);
    for i = 1:length(saved_results)
        combined_max_v_history(i) = saved_results(i).val_acc;
        combined_max_t_history(i) = saved_results(i).test_acc;
        combined_timing(i) = saved_results(i).timing(end);
    end
end                                   
%% optional: plot combined results
[combined_timing, index] = sort(combined_timing);
combined_max_v_history = combined_max_v_history(index);
combined_max_t_history = combined_max_t_history(index);

for i = 2:length(combined_timing)
    if combined_max_v_history(i-1) > combined_max_v_history(i)
        combined_max_v_history(i) = combined_max_v_history(i-1);
        combined_max_t_history(i) = combined_max_t_history(i-1);
    end
end
clf
hold on
if params.showAllHistory
    for i = 1:n_parallel
        plot(timings(i,1:lens(i)), max_v_histories(i,1:lens(i)), 'Color', [0.8 0.8 1]);
        plot(timings(i,1:lens(i)), max_t_histories(i,1:lens(i)), 'Color', [1 0.8 0.8]);
    end 
end
a = plot(combined_timing, combined_max_v_history, 'Color', [0 0 0.8], 'LineWidth', 2);
b = plot(combined_timing, combined_max_t_history, 'Color', [0.8 0 0], 'LineWidth', 2);
legend([a, b], 'val acc', 'test acc');
xlabel('time/s');
ylabel('accuracy');
hold off
%% return combined results
save(strcat('results/',name,'.mat'), 'saved_results', 'combined_max_v_history', 'combined_max_t_history', 'combined_timing'); 
savefig(strcat('results/figOf-',name,'.fig'));

combinedResult.combined_max_v_history = combined_max_v_history;
combinedResult.combined_max_t_history = combined_max_t_history;
combinedResult.combined_timing = combined_timing;
