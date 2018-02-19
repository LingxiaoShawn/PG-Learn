function [W, W_PreKNNExp] = SimGaussianGraph_KNN(X, k, a, Type, IsSparse, W_PreKNNExp, r, old_a_r)
% Construct a similarity graph and then sparsify it by KNN. 
% =========================================================
% Input:
%   X: n * d matrix input
%   k: param of KNN
%   a(A): vector or matrix, 
%      if a is vector A = diag(a) and all elements of a should greater than
%      0; if a is a matrix, A = a;   
%   Type: Type of similarity function and KNN type 
%       1 - Euclidean 
%       2 - Cosine (current not support)
%   IsSparse: Use sparse operation or not, if true then all following
%           operation in the algorithm(PG-Learn) will be sparse. 
%   W_PreKNNExp: -(diag(XAX) + diag(XAX)T - 2*XAX)
%   r: batch index
%   old_a_r: old a(r)
%
% Output: 
%   W: constructed graph (sparse matrix)
%   W_PreKNNExp: the part inside exp operation (for gaussian kernel) 
if nargin < 4
   ME = MException('InvalidCall:NotEnoughArguments', ...
       'Function called with too few arguments');
   throw(ME);
end
n = size(X,1);
%% calculate W_PreKNNExp
if nargin == 5
    %construct graph from scratch 
    W_PreKNNExp = 0;
else
    % construct graph based on old graph
    a = a(r) - old_a_r;
    X = X(:,r);
end
% transform X by a
if min(size(a)) == 1
    if size(a,1) ~= 1
        a = a';
    end
    XAX = bsxfun(@times, X, a) * X'; 
else
    XAX = X*a*X';
end
X_norm2 = diag(XAX);
% compute the input matrix of guassian kernel
if Type == 1 % Euclidean
    W_PreKNNExp = W_PreKNNExp + 2 * XAX - bsxfun(@plus, X_norm2, X_norm2');
end
if Type == 2 % Cosine (unfinished)
    W_PreKNNExp = W_PreKNNExp + XAX./sqrt(X_norm2*X_norm2') - 1;
end
%% create W 
if IsSparse
    % create sparse matrix
    % Preallocate memory
    indi = zeros(1, k * n);
    indj = zeros(1, k * n);
    inds = zeros(1, k * n);
    % sparsify by KNN
    for ii = 1:n   
        % sort row by distance
        [s, O] = sort(W_PreKNNExp(ii,:), 'descend');

        % Save indices and value of the k 
        indi((ii-1)*k+1 : ii*k) = ii;
        indj((ii-1)*k+1 : ii*k) = O(2:k+1); % ignore (i,i) item 
        inds((ii-1)*k+1 : ii*k) = s(2:k+1);
    end
    W = sparse(indi, indj, inds, n, n);
    clear indi indj inds dist s O;
    % exp 
    W = spfun(@exp, W);
else
    % create full matrix
    W = zeros(n,n);
    for ii = 1:n
        [s, O] = sort(W_PreKNNExp(ii,:), 'descend');
        W(ii,O(2:k+1)) = exp(s(2:k+1));
    end
end 
%% make it symmetric 
% symmetric W (mutual=min)
W = max(W, W');
end
