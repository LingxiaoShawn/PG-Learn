function [F, P] = LabelPropagation(W, Fl, alpha, max_iter)
% Semi-supervised learning label propagation of the algorithm in paper:
% Learning with Local and Global Consistency
% ==========================================
% Input:
%   W: n*n weight matrix.  The first L entries(row,col) are for labeled data,
%      the rest for unlabeled data.  W has to be symmetric, and all
%      entries has to be non-negative.  Also note the graph may be disconnected,
%      but each connected subgraph has to have at least one labeled point.
%      This is to make sure the sub-Laplacian matrix is invertible.
%   Fl: l*c label matrix.  Each line is for a labeled point, in 
%      one-against-all encoding (all zero but one 1). 
%   alpha: teleport probability, usually set as 0.99
%   max_iter: maximum number of iteration for transition, default 50
%
% Output: 
%   F: n*c label matrix.
%   P: normalized weight matrix, D^1/2 * W * D^1/2 

if nargin < 3
   alpha = 0.99;
   max_iter = 50;
end
% precision = 1e-8; % default precision

n = size(W,1); % total number of points
[l,c] = size(Fl); % the number of labeled points, the number of classes

% get Y
Y = sparse(n,c);
Y(1:l,:) = Fl;  % fl can be sparse

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
    % last_F = F;
    F = transit * F + base;
    % if max(max(abs(F-last_F))) < precision 
    %     break;
    % end
end

% do CMN or not? 

end