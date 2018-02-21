function params = SetConstantParams(params)
% set default params for: 
% params.graph_type  - 1=euclidean (current only support 1) 
% params.isSparse    - use sparse operation or not (suggest sparse)
% params.lambda      - learning rate for gradient-based method
% params.batch_size  - batch size for learning vector a
% params.k_range     - the random range for k of KNN graph
% params.split_val   - whether choose to  split labeled data into train set 
%                       + validation set
% params.tic_started - if the tic is started yet 
% params.method      - PG-Learn or MinEnt
if ~isfield(params, 'graph_type')
    params.graph_type = 1;
end
if ~isfield(params, 'isSparse')
    params.isSparse = true;
end
if ~isfield(params, 'lambda')
    params.lambda = 1;
end
if ~isfield(params, 'batch_size')
    params.batch_size = 50;
end
if ~isfield(params, 'k_range')
    params.k_range = [1,20];
end
if ~isfield(params, 'split_val')
    params.split_val = true;
end
if ~isfield(params, 'tic_started')
    params.tic_started = false;
end
if ~isfield(params, 'method')
    params.method = 'PG-Learn';
end