function v = vec(M)

[r,c] = size(M);
v = reshape(M,r*c,1);