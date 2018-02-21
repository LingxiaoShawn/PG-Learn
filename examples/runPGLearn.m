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
% set parameters
params_constant = struct();
params_constant = SetConstantParams(params_constant);
params_alterable.time = 120; %seconds
params_alterable.k = 5;
params_alterable.init_point = 1./(rand(1,dim) * (init_range(2) - init_range(1)) + init_range(1));
% run PG-Learn
result = PGLearn(data, params_constant, params_alterable)

