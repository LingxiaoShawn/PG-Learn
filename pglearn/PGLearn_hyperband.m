function combinedResult = RankGL_HParallel(data, dataInfoForNaming, params)
% Run RankGL in hyperband-parallel setting
% ===================================
% Inputs
%   data: the input data, XValid and yValid can be empty array []
%   | data.XTrain - n*d feature matrix for training data
%   | data.yTrain - n*1 class labels for training data (label starts from 0)
%   | data.XValid - n*d feature matrix for validation data
%   | data.yValid - n*1 class labels for validation data (label starts from 0)
%   | data.XTest - n*d feature matrix for test data
%   | data.yTest - n*1 class labels for test data (label starts from 0)
%   ------------------------------------------------------------------
%   dataInfoForNaming - a string contains all specific information of the data.
%   params: parameters for parallel setting as well as learning setting,
%   |     The user doesn't need to provide all of them, the lack input will use
%   |     default setting inside SetParallelParams method.
%   | params.nThreads - number of threads used 
%   | params.kRange - the range of k parameter(KNN)
%   | params.totalHours - number of time used
%   | params.rhoRange - the range of rho parameter(sigma = rho*sigma0)
%   | params.graphType - 1=euclidean 
%   | params.nRound - the number of round for hyperband setting
%   | params.halvingRate - the halving rate for hyperband setting
%   | params.timeMultiplier - the multiplier for increasing time of each round
%   | params.learningRate - learning rate for gradient-based method
%   | params.batchSize - batch size for learning vector a
%   | params.method - RankGL or RankMinEntGL
%   | params.isNewPool - whether start a new pool for the parallel run
%   ------------------------------------------------------------------
% Outputs
%   combinedResult: combined result for parallel setting.
%   | combinedResult.combined_max_v_history - validation accuracy
%   | combinedResult.combined_max_t_history - test accuracy
%   | combinedResult.combined_timing - corresponding time
%   | combinedResult.val_acc - best achieved validation accuracy
%   | combinedResult.test_acc - test accuracy w.r.t. val_acc
%   | combinedResult.a - learned a w.r.t. val_acc
%   -----------------------------------------------------
%% check parameters
if nargin < 2
    fprinf('Please give at least the first two args!');
    return;
end
if ~exist('params','var')
    params = struct();
end
params = SetParallelParams(params);
%% get parameters
mean_dist = mean(pdist([data.XTrain; data.XValid; data.XTest])); 
a_range = [(params.rhoRange(1)*mean_dist)^2, (params.rhoRange(2)*mean_dist)^2];
n_parallel = params.nThreads % number of threads
n_round = params.nRound;
k_range = params.kRange;
halving_rate = params.halvingRate;
time_multiplier = params.timeMultiplier;
total_hours = params.totalHours;
start_new_pool = params.isNewPool;

params_constant.lambda = params.learningRate;
params_constant.batch_size = params.batchSize;
params_constant.graph_type = params.graphType;
params_constant.isSparse = params.isSparse;
params_constant.method = params.method;

base_time = total_hours * 3600 / (time_multiplier^n_round - 1); % total seconds
a_dim = size(data.XTrain, 2); % dimension of a
n_assigned_starts = fix(n_parallel/2); 
n_rand_starts = n_parallel - n_assigned_starts;
assigned_starts = linspace(a_range(1),a_range(2),n_assigned_starts)';
%% Set cluster and start a pool 
clust = parcluster('local');
clust.NumWorkers = n_parallel + 1;
if start_new_pool
    delete(gcp('nocreate'));
end
p = gcp('nocreate');
if isempty(p)
    parpool(clust, clust.NumWorkers);
else
    n_parallel = p.NumWorkers - 1;
end
%% history init
max_v_histories = [];
max_t_histories = [];
max_g_histories = [];
timings = [];
gs = [];
ori_val_accs = [];
ori_test_accs = [];

lengths = [];
combined_max_v_history = [];
combined_max_t_history = [];
combined_timing = [];
lens_detail = [];

converged_index = 1;
converged_max_v_new = 0;
converged_max_t_new = 0;
converged_g_new = 0;
converged_i_new = 0;
%% parallel local search
spmd
    % use the first worker as master worker
    if labindex == 1
        % init parameters
        fprintf('master is working...\n');
        n_halved = fix(n_parallel * halving_rate);
        n_succeeded = n_parallel - n_halved;
        
        % hyperparameter initialization
        runtime = base_time;
        time_base = 0;
        init_points = zeros(n_parallel, a_dim);
        % assign starts
        init_points(1:n_assigned_starts, :) = repmat(assigned_starts,1,a_dim);
        % rand starts
        init_points(n_assigned_starts + 1:end, :) = 1./(rand(n_rand_starts,a_dim) * (a_range(2) - a_range(1)) + a_range(1));            
        start_rounds = ones(1,n_parallel);
        % assign random k
        ks = randi(k_range, 1, n_parallel);
        
        for i = 1:n_round
            fprintf('Round: %d\n',i);
            % Step 1: master send alterable parameters to slaves
            for j = 2:numlabs
                send_data.time = runtime;
                send_data.init_point = init_points(j-1,:);
                send_data.k = ks(j-1);
                labSend(send_data, j);
            end      
            % Step 2: recieve result from all slaves, then do successive
            % halving, and reassign some starting points
            for j = 2:numlabs
                results(j-1) = labReceive(j);
            end
            %% append results of each round to the history of all rounds
            lens = zeros(1,n_parallel);
            for j = 1:n_parallel
                lens(j) = length(results(j).timing);
            end
            max_len = max(lens);      
            % update converged_max_v and converged_max_t
            converged_max_v = converged_max_v_new;
            converged_max_t = converged_max_t_new;
            converged_g = converged_g_new;
            converged_i = converged_i_new; 
            % process results of converged threads
            for j = 1:n_parallel
                if results(j).converged
                    if converged_max_v_new < results(j).val_acc
                        converged_max_v_new = results(j).val_acc;
                        converged_max_t_new = results(j).test_acc;
                        converged_g_new = results(j).max_g_history(end);
                        converged_i_new = converged_index;
                    elseif converged_max_v_new == results(j).val_acc
                        % prevent overfit, choose the one with largest loss
                        if results(j).max_g_history(end) > converged_g_new
                            converged_max_t_new = results(j).test_acc;
                            converged_g_new = results(j).max_g_history(end);
                            converged_i_new = converged_index;
                        end
                    end
                    % save the result to the pool of all converged results                
                    converged_results(converged_index).val_acc = results(j).val_acc;
                    converged_results(converged_index).test_acc = results(j).test_acc;
                    converged_results(converged_index).init_point = results(j).init_point;
                    converged_results(converged_index).start_round = start_rounds(j);
                    converged_results(converged_index).converged_round = i;
                    converged_results(converged_index).g = results(j).max_g_history(end);
                       
                    converged_index = converged_index + 1;
                    % expand the real-time histories of the converged thread
                    n_expand = max_len - lens(j);
                    results(j).timing = [results(j).timing, linspace(results(j).time_cost, runtime, n_expand)];
                    results(j).max_v_history = [results(j).max_v_history, ones(1,n_expand)*results(j).max_v_history(end)];
                    results(j).max_t_history = [results(j).max_t_history, ones(1,n_expand)*results(j).max_t_history(end)];
                    results(j).max_g_history = [results(j).max_g_history, ones(1,n_expand)*results(j).max_g_history(end)];
                    results(j).g = [results(j).g, ones(1,n_expand)*results(j).g(end)];
                    results(j).ori_val_acc = [results(j).ori_test_acc, ones(1, n_expand)*results(j).ori_val_acc(end)];
                    results(j).ori_test_acc = [results(j).ori_test_acc, ones(1,n_expand)*results(j).ori_test_acc(end)];
                    lens(j) = max_len;
                    % set val_acc to 0 to help halving procedure
                    results(j).val_acc = 0;
                end
            end
            
            % process results of all threads 
            max_v_his_temp = zeros(n_parallel, max_len);
            max_t_his_temp = zeros(n_parallel, max_len);
            max_g_his_temp = zeros(n_parallel, max_len);
            timings_temp = zeros(n_parallel, max_len);
            gs_temp = zeros(n_parallel, max_len);
            ori_val_accs_temp = zeros(n_parallel, max_len);
            ori_test_accs_temp = zeros(n_parallel, max_len);
                   
            for j = 1:n_parallel
                max_v_his_temp(j, 1:lens(j)) = results(j).max_v_history;
                max_t_his_temp(j, 1:lens(j)) = results(j).max_t_history;
                timings_temp(j, 1:lens(j)) = results(j).timing;
                max_g_his_temp(j, 1:lens(j)) = results(j).max_g_history;
                gs_temp(j, 1:lens(j)) = results(j).g;
                ori_val_accs_temp(j, 1:lens(j)) = results(j).ori_val_acc;
                ori_test_accs_temp(j, 1:lens(j)) = results(j).ori_test_acc;     
            end
            timings_temp = timings_temp + time_base;
            
            % combine 32 histories to 1 according to val_acc and g: first
            % for all ith results, find the threads that have largest
            % val_acc. If there is only one thread achieves highest val_acc
            % then use this result, otherwise comparing g value among all
            % threads achieving highest val_acc and use the one with highest
            % g.(the one with highest g has more potential to improve more)      
            combined_max_v_his_temp = zeros(1, max_len);
            combined_max_t_his_temp = zeros(1, max_len);
            combined_timing_temp = zeros(1, max_len);
      
            for j = 1:max_len
                combined_max_v_his_temp(j) = max(max_v_his_temp(:,j));
                temp_index = find(max_v_his_temp(:,j) == combined_max_v_his_temp(j));
                if length(temp_index) > 1
                    % multiple threads achieve max validation history.
                    % choose the one with largest g value.
                    [g_value, i_g] = max(max_g_his_temp(temp_index,j));
                    combined_max_t_his_temp(j) = max_t_his_temp(temp_index(i_g),j);
                    combined_timing_temp(j) = timings_temp(temp_index(i_g),j);        
                else
                    combined_max_t_his_temp(j) = max_t_his_temp(temp_index(1),j);
                    combined_timing_temp(j) = timings_temp(temp_index(1),j);
                    g_value = max_g_his_temp(temp_index(1),j);
                end
                % considering converged results
                if combined_max_v_his_temp(j) < converged_max_v
                    combined_max_v_his_temp(j) = converged_max_v;
                    combined_max_t_his_temp(j) = converged_max_t;
                elseif combined_max_v_his_temp(j) == converged_max_v 
                    if g_value ~= 0 && g_value < converged_g
                        combined_max_v_his_temp(j) = converged_max_v;
                        combined_max_t_his_temp(j) = converged_max_t;
                    end
                end
            end
            % append result of this round to all histories
            max_v_histories = [max_v_histories, max_v_his_temp];
            max_t_histories = [max_t_histories, max_t_his_temp];
            max_g_histories = [max_g_histories, max_g_his_temp];
            timings = [timings, timings_temp];
            
            gs = [gs, gs_temp];
            ori_val_accs = [ori_val_accs, ori_val_accs_temp];
            ori_test_accs = [ori_test_accs, ori_test_accs_temp];
                    
            combined_max_v_history = [combined_max_v_history, combined_max_v_his_temp];
            combined_max_t_history = [combined_max_t_history, combined_max_t_his_temp];
            combined_timing = [combined_timing, combined_timing_temp];
            lengths = [lengths, max_len];
            lens_detail = [lens_detail, lens'];
            %% halving policy (sucessive halving)
            time_base = time_base + runtime;
            runtime = runtime * time_multiplier;
            val_accs = 1:n_parallel;
            for j = 1:n_parallel
                val_accs(j) = results(j).val_acc;
            end
            [val_accs, val_order] = sort(val_accs,'descend');
            results = results(val_order);
            for j = 1:n_succeeded
                init_points(j, :) = results(j).init_point;
            end
            init_points(n_succeeded + 1: end, :) = 1./(rand(n_halved,a_dim) * (a_range(2) - a_range(1)) + a_range(1)); 
            start_rounds = start_rounds(val_order);
            ks = ks(val_order);
            if i < n_round
                start_rounds(n_succeeded + 1: end) = i + 1;
                ks(n_succeeded + 1: end) = randi(k_range, 1, n_halved);
            end
        end
        %% add additional information into results
        for j = 1:n_parallel
            results(j).k = ks(j);
            results(j).start_round = start_rounds(j);
        end
    else
    % all other workers are slaves
        fprintf('Worker %d is working...\n',labindex);
        for i = 1:n_round
            % step 1: slaves receive arguments from, then run local_search,
            % return max_val_acc and ending_point
            params_received = labReceive(1);
            result = PGLearn(data, params_constant, params_received);
            % step 2: send max_val_acc and ending_point back to master
            labSend(result, 1);
        end       
    end    
end 
%% get result
output_saved = results{1};
if exist('converged_results','var')
    converged_results = converged_results{1};
end
name = strcat(dataInfoForNaming, params.method, ...
        sprintf('_HParallel_%dtasks_%gh', n_parallel, total_hours)); 
%% plot maximum accuracy
max_v_histories = max_v_histories{1};
max_t_histories = max_t_histories{1};
timings = timings{1};
lengths = lengths{1};
combined_max_v_history = combined_max_v_history{1};
combined_max_t_history = combined_max_t_history{1};
combined_timing = combined_timing{1};
% sort combined results according to time
[combined_timing, index] = sort(combined_timing);
combined_max_v_history = combined_max_v_history(index);
combined_max_t_history = combined_max_t_history(index);

% process combined history
for i = 2:length(combined_timing)
    if combined_max_v_history(i-1) > combined_max_v_history(i)
        combined_max_v_history(i) = combined_max_v_history(i-1);
        combined_max_t_history(i) = combined_max_t_history(i-1);
    end
end

lens_detail = lens_detail{1};

% save all histories
clear all_histories
all_histories.max_v_histories = max_v_histories;
all_histories.max_t_histories = max_t_histories;
all_histories.timings = timings;
all_histories.lengths = lengths;
all_histories.lens_detail = lens_detail;
all_histories.combined_max_v_history = combined_max_v_history;
all_histories.combined_max_t_history = combined_max_t_history;
all_histories.combined_timing = combined_timing;
all_histories.max_g_histories = max_g_histories{1}; 
all_histories.gs = gs{1};
all_histories.ori_val_accs = ori_val_accs{1};
all_histories.ori_test_accs = ori_test_accs{1};

clf
hold on
for i = 1:n_parallel
    index_base = 0;
    for j = 1:length(lengths)
        plot(timings(i,index_base+1:index_base+lens_detail(i,j)), max_v_histories(i,index_base+1:index_base+lens_detail(i,j)), 'Color', [0.8 0.8 1]);
        plot(timings(i,index_base+1:index_base+lens_detail(i,j)), max_t_histories(i,index_base+1:index_base+lens_detail(i,j)), 'Color', [1 0.8 0.8]);
        index_base = index_base + lengths(j);
    end
end
a = plot(combined_timing, combined_max_v_history, 'Color', [0 0 0.8], 'LineWidth', 2);
b = plot(combined_timing, combined_max_t_history, 'Color', [0.8 0 0], 'LineWidth', 2);
legend([a, b], 'val acc', 'test acc');
xlabel('time/s');
ylabel('accuracy');
hold off
%% get a, val_acc, test_acc from output_saved and converged_results
val_acc = 0;
test_acc = 0;
g_value = 0;
a = 0;
for i = 1:length(output_saved)
    if val_acc < output_saved(i).max_v_history(end)
        val_acc = output_saved(i).max_v_history(end);
        test_acc = output_saved(i).max_t_history(end);
        g_value = output_saved(i).max_g_history(end);
        a = output_saved(i).init_point;
    elseif val_acc == output_saved(i).max_v_history(end)
        if g_value < output_saved(i).max_g_history(end)
            val_acc = output_saved(i).max_v_history(end);
            test_acc = output_saved(i).max_t_history(end);
            g_value = output_saved(i).max_g_history(end);
            a = output_saved(i).init_point;
        end
    end
end

if exist('converged_results','var')
    best_converged_result = converged_results(converged_i_new{1});
    if val_acc < best_converged_result.val_acc
        val_acc = best_converged_result.val_acc;
        test_acc = best_converged_result.test_acc;
        a = best_converged_result.init_point;
    elseif val_acc == best_converged_result.val_acc
        if g_value <= best_converged_result.g
            val_acc = best_converged_result.val_acc;
            test_acc = best_converged_result.test_acc;
            a = best_converged_result.init_point;
        end
    end   
end
%% save to disk
savefig(strcat('results/parallel/figOf-',name,'.fig'));
if exist('converged_results','var')
    save(strcat('results/parallel/',name,'.mat'), 'output_saved', 'converged_results', 'all_histories');
else
    save(strcat('results/parallel/',name,'.mat'), 'output_saved', 'all_histories');
end
%% return 
combinedResult.combined_max_v_history = combined_max_v_history;
combinedResult.combined_max_t_history = combined_max_t_history;
combinedResult.combined_timing = combined_timing;
combinedResult.val_acc = val_acc;
combinedResult.test_acc = test_acc;
combinedResult.a = a;
