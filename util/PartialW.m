function [partialW, S] = PartialW(X_batch, W, Type)
% Calculate derivative of W with respect to a.
% ============================================
% Input:
%   X_batch: n*batch_size matrix, X(:, batch_indexes). 
%   W: constructed graph, if W is sparse, then partial W will be computed
%       sparsely.
%   Type: Type of similarity function and KNN type 
%       1 - Euclidean 
%       2 - Cosine (current not support)
%
% Output:
%   partialW: n*n*batch_size 3d matrix, partial W with respect to a
%   S: n*n*batch_size 3d matrix,  W*1 * (partialW*1)T, thia will be used of
%   calculating partialP

[n, batch_size] = size(X_batch);

if ~issparse(W)
    %% full space operation
    X_batch = reshape(X_batch, [n, 1, batch_size]);
    X2 = X_batch.^2;
    if Type == 1
        deltaX = repmat(X2, 1,n,1) + repmat(permute(X2, [2 1 3]), n,1,1) - 2 * mtimesx(X_batch,X_batch,'t');
        partialW = repmat(-W, 1,1,batch_size).* deltaX;
        % deltaX = bsxfun(@plus, X2, permute(X2, [2,1,3])) - 2 * mtimesx(X_batch,X_batch,'t');
        % partialW = bsxfun(@times, -W, deltaX);
    end  
    S = mtimesx(sum(W, 2), permute(sum(partialW, 2), [2 1 3])); % n*n*batch_size 3d matrix
else
    %% use n*n*d 3d sparse matrix (pretty fast and memory-saving)
    % get the sparse index of W
    [row,col] = find(W);
    % get number of nonzero elements
    nnz = size(row,1);
    % get all value of sparse matrix
    val_deltaX = (X_batch(row,:) - X_batch(col,:)).^2; 
    % construct index of new sparse matrix(use 1d first)
    index = bsxfun(@plus, find(W),repmat((0:batch_size-1)*n^2, nnz, 1));
    % get partialW
    deltaX = sparse(index, 1, val_deltaX, batch_size*n^2, 1); % n*nd 2d sparse matrix 
    deltaX = ndSparse(deltaX, [n,n,batch_size]); % n*n*d 3d sparse matrix 
    partialW = bsxfun(@times, -W, deltaX);
    % compute S
    Wdot1 = sum(W,2);
    partialWdot1T = reshape(sum(partialW, 2),[n,batch_size]);
    val_S = full(bsxfun(@times, Wdot1(row), partialWdot1T(col,:)));
    S = sparse(index, 1, val_S, batch_size*n^2, 1); % n*nd 2d sparse matrix
    S = ndSparse(S,[n,n,batch_size]); % n*n*d 3d sparse matrix
end
end