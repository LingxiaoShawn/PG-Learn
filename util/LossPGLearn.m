function [g, partialG] = LossPGLearn(Fv, partialFv, yVal)
% Calculate loss of learning-to-rank objective function g and its graident 
% with respect to a according to: 
%       g =  1/n * sum_c sum_uv [-log(puv_c)]
%       partialG = 1/n * sum_c sum_uv [ (puv_c - 1)*(partialFu_c - partialFv_c)]
% Where u belongs to validating x with label c, v belongs to validaing x
% without label c, and n is the number of used samples. 
% =====================================================
% Input:
%   Fv/Fl: 
%          (l0-l)*c matrix, predicted label matrix (only valiadtion part);
%           or l0*c matrix, predicted F matrix on labeled data part.
%           where l 0 is the number of labeled sample, l is the number of  training data.
%   partialFv/partialFl: 
%          (l0-l)*c*batch_size matrix, derivative of F with respect to a (only validation part);
%           or l0*c*batch_size matrix (labeled data part).
%   yVal: ground true label of validation set/labeled set. 
% Output: 
%   g: scalar, value of learning-to-rank objective function 
%   partialG: vector, derivative of g with respect to a
 
n_class = size(Fv,2);
g = 0;
partialG = 0;

for c = 0 : n_class - 1
    p_c = sigmf(bsxfun(@minus, Fv(yVal == c, c + 1), ...
                Fv(yVal ~= c, c + 1)'), [1 0]);
    g = g + sum(sum(-log(p_c+1e-100)));
    delta_partial_F = bsxfun(@minus, partialFv(yVal == c, c + 1, :), ...
                             permute(partialFv(yVal ~= c, c + 1, :), [2 1 3]));
    partialG = partialG + reshape(sum(sum(bsxfun(@times, p_c - 1, delta_partial_F))),1,[]);
end

% average loss function and gradient 
g = g / size(Fv, 1);
partialG = partialG / size(Fv, 1);

end