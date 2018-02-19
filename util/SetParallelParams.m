function params = SetParallelParams(params)
% set default params for: 
% params.nThreads       - number of threads used for parallel
% params.kDivider       - for grid search, the divider for k range
% params.kRange         - the random range for k of KNN graph 
% params.totalHour      - total runtime, budget 
% params.rhoRange       - the random range for rho, rho is used by 
%                             sigma = rho * sigma_0
% params.graphType      - 1=euclidean (current only support 1) 
% params.isSparse       - use sparse operation or not (suggest sparse)
% params.nRound         - the number of round for hyperband setting
% params.halvingRate    - the halving rate for hyperband setting
% params.timeMultiplier - the multiplier for increasing time of each round
% params.learningRate   - learning rate for gradient-based method
% params.batchSize      - batch size for learning vector a 
TODO: % params.method         - PG-Learn or MinEnt
% params.isNewPool      - whether start a new pool for the parallel run
% params.showAllHistory - whether show all history, this is for RndSet
%                         parallel framework
if ~isfield(params, 'nThreads')
    params.nThreads = 32;
end
if ~isfield(params, 'kDivider')
    params.kDivider = 4;
end
if ~isfield(params, 'totalHours')
    params.totalHours = 2;
end
if ~isfield(params, 'rhoRange')
    params.rhoRange = [0.1, 10];
end
if ~isfield(params, 'kRange')
    params.kRange = [1, 20];
end
if ~isfield(params, 'graphType')
    params.graphType = 1;
end
if ~isfield(params, 'isSparse')
    params.isSparse = true;
end
if ~isfield(params, 'nRound')
    params.nRound = 6;
end
if ~isfield(params, 'halvingRate')
    params.halvingRate = 1/2;
end
if ~isfield(params, 'timeMultiplier')
    params.timeMultiplier = 2;
end
if ~isfield(params, 'learningRate')
    params.learningRate = 1;
end
if ~isfield(params, 'batchSize')
    params.batchSize = 50;
end
if ~isfield(params, 'method')
    params.method = 'PG-Learn';
end
if ~isfield(params, 'isNewPool')
    params.isNewPool = true;
end
if ~isfield(params, 'showAllHistory')
    params.showAllHistory = false;
end


