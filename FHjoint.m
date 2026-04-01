function E = FHjoint(R1, R2)
%FHJOINT  Fuzzy joint information measure between two fuzzy relations
%   Inputs:
%       R1 : n-by-n fuzzy relation matrix
%       R2 : n-by-n fuzzy relation matrix
%
%   Output:
%       E  : scalar fuzzy joint information measure

    % Basic size checks
    if ~isequal(size(R1), size(R2))
        error('FHjoint:SizeMismatch', 'R1 and R2 must have the same size.');
    end

    [n, m] = size(R1);
    if n ~= m
        warning('FHjoint:NonSquare', ...
            'R1 and R2 are not square; interpretation assumes n-by-n relations.');
    end

    % Fuzzy cardinalities
    card1 = sum(R1, 2).';    % 1-by-n
    card2 = sum(R2, 2).';    % 1-by-n

    % Fuzzy intersection using min operator
    Rint     = min(R1, R2);      % n-by-n
    cardInt  = sum(Rint, 2).';   % 1-by-n

    % Compute the term inside the log2
    num = n*ones(1,n);
    tmp = num.*cardInt./(card1.*card2);

    % Final joint information measure
    E = (1 / n) * sum(log2(tmp));


end