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
params_constant = struct();
params_constant = SetConstantParams(params_constant);
params_constant.split_val = false;
params_alterable.a_range = init_range;
params_alterable.init_point = 1./(ones(1,dim)*rand() * (init_range(2) - init_range(1)) + init_range(1));
params_alterable.time = 120; % seconds
params_alterable.k_range = [1,20];
params_alterable.k = 5; 
params_alterable.sigma_range = [0.3162 * mean_dist 3.1623 * mean_dist];
%% random search
result = random_search(data, params_constant, params_alterable)
%% grid search 
result = grid_search(data, params_constant, params_alterable)
%% AEW 
result = AEW_LLGC(data, params_constant, params_alterable)
%% IDML
result = IDML_ITML(data, params_constant, params_alterable)
%% MinEnt
params_constant.method = 'MinEnt';
result = PGLearn(data, params_constant, params_alterable)
%% PG-Learn
params_constant.method = 'PG-Learn';
result = PGLearn(data, params_constant, params_alterable)
