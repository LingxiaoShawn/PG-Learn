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
dataInfoString = sprintf('10%%noise_level%s_partition%d_trainPer%d%%_noise%d',...
                                 dataset, partition, train_per*100, noise_level);
% set parameters
params_paralllel = struct();
params_parallel = SetParallelParams(params_parallel);
params_parallel.totalHours = 0.25;
params_parallel.isNewPool = false;
result = PGLearn_hyperband(data, dataInfoString, params_parallel);
