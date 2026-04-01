function R = simR(f1, f2)

%SIMR  Fuzzy similarity matrix for numeric targets.
%
%   R = SIMR(f1, f2) returns an n1-by-n2 similarity matrix with
%   elements
%       R(i,j) = exp(-abs(f1(i) - f2(j))).
%
%   Inputs:
%       f1 : n1-by-1 (or 1-by-n1) vector
%       f2 : n2-by-1 (or 1-by-n2) vector
%
%   Output:
%       R  : n1-by-n2 matrix of similarities in [0,1]

n = length(f1);
for i=1:n
    for j=1:n
        R(i,j)=exp(-abs(f1(i)-f2(j)));
    end
end


end