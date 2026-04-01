function R = simL(f1, f2, p)
%SIML  Łukasiewicz-based fuzzy similarity between two numeric vectors.
%
%   R = SIML(f1, f2, p) returns an n1-by-n2 similarity matrix 
%
%   Inputs:
%       f1 : n1-by-1 (or 1-by-n1) numeric vector
%       f2 : n2-by-1 (or 1-by-n2) numeric vector
%       p  : (optional) positive scalar parameter, default p = 1
%
%   Output:
%       R  : n1-by-n2 similarity matrix

% Default p if not provided
if nargin<3
    p=1;
end

n=length(f1);
for i=1:n
    for j=1:n
        R(i,j)=(1-abs(f1(i)^p-f2(j)^p))^(1/p);

    end
end


end