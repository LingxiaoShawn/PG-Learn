function combinedResult = random_search_parallel(data, dataInfoForNaming, params)
% Run random search in parallel setting
% ===================================
% Inputs
%   data: processed data, divided into labeled data and unlabeled data.
%   | data.XTrain - n*d feature matrix for training data
%   | data.yTrain - n*1 class labels for training data (label starts from 0)
%   | data.XValid - n*d feature matrix for validation data
%   | data.yValid - n*1 class labels for validation data (label starts from 0)
%   | data.XTest - n*d feature matrix for test data
%   | data.yTest - n*1 class labels for test data (label starts from 0) 
%   -------------------------------------------------------------------
%   dataInfoForNaming: a string contains all specific information of the data.
%   params: parameters for parallel setting as well as learning setting.
%   | params.nThreads - number of threads used 
%   | params.kRange - the range of k parameter(KNN)
%   | params.totalHours - number of time used
%   | params.rhoRange - the range of rho parameter(sigma = rho*sigma0)
%   | params.graphType - 1=euclidean 
%   | params.isSparse - use sparse operation or not (suggest sparse)
%   | params.isNewPool - whether start a new parallel pool
%   --------------------------------------------------------------------
% Outputs
%   combinedResult: combined result for parallel setting.
%   | combinedResult.combined_max_v_history - validation accuracy
%   | combinedResult.combined_max_t_history - test accuracy
%   | combinedResult.combined_timing - corresponding time
%   --------------------------------------------------------------------
%% check parameters
if nargin < 2
    fprinf('Please give at least the first two args!');
    return;
end
if ~exist('params','var')
    params = struct();
end
params = SetParallelParams(params);
%% get search settings
mean_dist = mean(pdist([data.XTrain; data.XValid; data.XTest])); 
a_range = [(params.rhoRange(1)*mean_dist)^2, (params.rhoRange(2)*mean_dist)^2];
total_hours = params.totalHours;
n_parallel = params.nThreads;
total_runtime = n_parallel*total_hours*3600; % in seconds

params_constant.k_range = params.kRange;
params_constant.graph_type = params.graphType; % 1 = euclidean
params_constant.isSparse = params.isSparse;

start_new_pool = params.isNewPool;
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
    n_parallel = p.NumWorkers - 1
end
%% run with #threads = n_parallel
single_runtime = total_runtime / n_parallel;
spmd 
    % master
    if labindex == 1
        send_data.time = single_runtime;
        send_data.a_range = a_range;
        for j = 2:numlabs
            labSend(send_data, j);
        end
        for j = 2:numlabs
            % results(j-1) = receive_data from worker j
            results(j-1) = labReceive(j);    
        end
    % slaves    
    else
        parameters = labReceive(1);
        result = random_search(data, params_constant, parameters);
        labSend(result, 1);
    end
end

%% show/save results
saved_results = results{1};
name = strcat(dataInfoForNaming, sprintf('_RS_parallel_%dtasks_%gh',...
                                          n_parallel, total_hours));
%% plot max_v_history and max_t_history, with respect to timing
lens = zeros(1,n_parallel);
for i = 1:n_parallel
    lens(i) = length(saved_results(i).timing);
end
max_len = max(lens);
min_len = min(lens);
max_v_histories = zeros(n_parallel, max_len);
max_t_histories = zeros(n_parallel, max_len);
timings = zeros(n_parallel, max_len);

for i = 1:n_parallel
    max_v_histories(i, 1:lens(i)) = saved_results(i).max_v_history;
    max_t_histories(i, 1:lens(i)) = saved_results(i).max_t_history;
    timings(i, 1:lens(i)) = saved_results(i).timing;
end

combined_max_v_history = 1:min_len;
combined_max_t_history = 1:min_len;
combined_timing = 1:min_len;
for i = 1:min_len
    [combined_max_v_history(i), index] = max(max_v_histories(:,i));
    combined_max_t_history(i) = max_t_histories(index, i);
    combined_timing(i) = timings(index, i);
end

% sort combined results according to time
[combined_timing, index] = sort(combined_timing);
combined_max_v_history = combined_max_v_history(index);
combined_max_t_history = combined_max_t_history(index);

clf
hold on
for i = 1:n_parallel
    plot(timings(i, 1:lens(i)), max_v_histories(i, 1:lens(i)), 'Color', [0.8 0.8 1]);
    plot(timings(i, 1:lens(i)), max_t_histories(i, 1:lens(i)), 'Color', [1 0.8 0.8]);
end
a = plot(combined_timing, combined_max_v_history, 'Color', [0 0 0.8], 'LineWidth', 2);
b = plot(combined_timing, combined_max_t_history, 'Color', [0.8 0 0], 'LineWidth', 2);
legend([a, b], 'RS val acc', 'RS test acc');
xlabel('time/s');
ylabel('accuracy');
hold off
%% save 
savefig(strcat('results/figOf-',name,'.fig'));
save(strcat('results/',name,'.mat'), 'saved_results','combined_max_v_history','combined_max_t_history', 'combined_timing');
%% return
combinedResult.combined_max_v_history = combined_max_v_history;
combinedResult.combined_max_t_history = combined_max_t_history;
combinedResult.combined_timing = combined_timing;
