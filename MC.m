function R = MC(class)
% MC  Compute crisp fuzzy relation matrix for class labels.
%
%   R = MC(class) returns an n-by-n binary relation matrix where
%   R(i,j) = 1 if class(i) == class(j), and 0 otherwise.
%
%   Input:
%       class : n-by-1 vector of class labels
%
%   Output:
%       R     : n-by-n relation matrix (1 = same class, 0 = different)

n = length(class);

for i=1:n
    for j=1:n
        if class(i)==class(j)
           R(i,j)=1;
        else
           R(i,j)=0;
        end
    end
end

%class = class(:);            
%R = double(class == class'); % vectorized comparison (fast!)

end
