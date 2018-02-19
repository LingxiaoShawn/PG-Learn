function [H, partialH] = LossMinEnt(Fu, partialFu)
% Calculate loss of MinEnt objective function H and its graident 
% with respect to a according to: (without doing CMN)
%       H = -1/u \sum_i \sum_j Fu_ij * log(Fu_ij)
%       partialH = -1/u \sum_i \sum_j partialFu_ij *(log(Fu_ij) + 1)
% Where u belongs to validating x with label c, v belongs to validaing x
% without label c. 
% ================
% Input:
%   Fu: (n-l0)*c matrix, predicted label matrix (only unlabeled part)
%   partialFu: (n-l0)*c*batch_size matrix, derivative of F with respect to
%               a (only unlabeled part)
% Output: 
%   H: scalar, value of learning-to-rank objective function 
%   partialH: vector, derivative of g with respect to a

u = size(Fu, 1);
logFu = log(Fu + 1e-100);
H = -1/u * sum(sum(Fu.*logFu));
partialH = -1/u * reshape(sum(sum(bsxfun(@times, logFu + 1 , partialFu))), 1,[]);

end