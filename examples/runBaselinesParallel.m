% load data
dataset = 'mnist';
noise_level = 0;
train_per = 0.2;
partition = 1;
data = LoadData(dataset, noise_level, train_per, partition);
X = [data.XTrain; data.XValid; data.XTest];
mean_dist = mean(pdist(X));
dim = size(X,2);
init_range = [0.1 * mean_dist^2, 10 * mean_dist^2];
rng('shuffle');
% set params
dataInfoString = sprintf('%s_partition%d_trainPer%d%%_noise%d',...
                                 dataset, partition, train_per*100, noise_level);
params_parallel = struct();
params_parallel = SetParallelParams(params_parallel);
params_parallel.totalHours = 0.25;
params_parallel.isNewPool = false;
%% MinEnt
params_parallel.param1Set = 1:20;
params_parallel.param2Set = 0.1:0.05:10;
params_parallel.showAllHistory = true;
dataInfoStringN = strcat(dataInfoString, 'MinEnt');
result = RndSet_parallel_framework(@PGLearn, data, dataInfoStringN, params_parallel);
%% GS
result = grid_search_parallel(data, dataInfoString, params_parallel);
%% RS
result = random_search_parallel(data, dataInfoString, params_parallel);
%% IDML
params_parallel.showAllHistory = false; % for IDML and AEW not support true option
params_parallel.param3Set = [0.05 0.1 0.15 0.2];
dataInfoStringN = strcat(dataInfoString, 'IDML');
result = RndSet_parallel_framework(@IDML_ITML, data, dataInfoStringN, params_parallel);
%% AEW 
dataInfoStringN = strcat(dataInfoString, 'AEW');
result = RndSet_parallel_framework(@AEW_LLGC, data, dataInfoStringN, params_parallel);
