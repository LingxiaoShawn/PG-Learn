function [F, P] = LabelPropagation(W, fl, alpha, max_iter)
% semi-supervised learning label propagation
%
% Input:
%   W: n*n weight matrix.  The first L entries(row,col) are for labeled data,
%      the rest for unlabeled data.  W has to be symmetric, and all
%      entries has to be non-negative.  Also note the graph may be disconnected,
%      but each connected subgraph has to have at least one labeled point.
%      This is to make sure the sub-Laplacian matrix is invertible.
%   fl: l*c label matrix.  Each line is for a labeled point, in 
%      one-against-all encoding (all zero but one 1).  For example in binary
%      classification each line would be either "0 1" or "1 0".
%   alpha: teleport probability
%   max_iter: maximum number of iteration for transition
%
% Output: 
%   F: n*c label matrix.
%   P: normalized weight matrix, D^1/2 * W * D^1/2 

precision = 1e-8; % default precision

n = size(W,1); % total number of points
[l,c] = size(fl); % the number of labeled points, the number of classes

% get Y
Y = sparse(n,c);
Y(1:l,:) = fl;  % fl can be sparse

% compute normalized transition matrix P
degs = sum(W,2);
degs(degs == 0) = 1e-13;
D = spdiags(1./(degs.^0.5), 0, n, n);
P = D * W * D;

% iterative solution of label propagation
F = Y;
base = (1 - alpha) * Y;
transit = alpha * P;
for  i = 1:max_iter
%    last_F = F;
    F = transit * F + base;
%     if max(max(abs(F-last_F))) < precision 
%         i
%         break;
%     end
end

% do CMN or not? 

end