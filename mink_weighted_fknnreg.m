function predicted = mink_weighted_fknnreg(xtrain, ytrain, xtest, k_values, fuzzy, m, p, w)
%MINK_WEIGHTED_FKNNREG  Fuzzy k-NN regression with weighted Minkowski distance.
%
%   predicted = mink_weighted_fknnreg(xtrain, ytrain, xtest, k_values, fuzzy, m, p, w)
%
%   Inputs:
%       xtrain   : nTrain x d  matrix of training features
%       ytrain   : nTrain x 1  vector 
%       xtest    : nTest  x d  matrix of test features
%       k_values : 1 x K  vector of k values (e.g., [1 3 5])
%       fuzzy    : (optional) logical, true for fuzzy kNN, false for ordinary kNN
%                  default = true
%       m        : (optional) fuzzifier parameter (>1), default = 2
%       p        : (optional) Minkowski distance order, default = 2 (Euclidean)
%       w        : (optional) 1 x d vector of non-negative feature weights
%                  if not provided, all features are weighted equally

%   Output:
%       predicted : nTest x length(k_values) matrix of predictions,
%                   each column corresponds to a k in k_values.

%   Created by Mahinda Mailagaha Kumbure & Pasi Luukka

    % -----------------------------
    % Handle defaults
    % -----------------------------
    if nargin < 5 || isempty(fuzzy)
        fuzzy = true;
    end

    if nargin < 6 || isempty(m)
        m = 2;
    end

    if nargin < 7 || isempty(p)
        p = 2;
    end

    [~, nFeat] = size(xtrain);
    nTest      = size(xtest, 1);

    if nargin < 8 || isempty(w)
        w = ones(1, nFeat);
    end

    % Ensure weight vector has correct size
    if numel(w) ~= nFeat
        error('fknnreg_mink_weighted:WeightSizeMismatch', ...
              'Length of w (%d) does not match number of features (%d).', ...
               numel(w), nFeat);
    end

    % -----------------------------
    % Precompute weighted training data
    % -----------------------------
    % We use sqrt(w) because Minkowski distance involves |x - z|^p;
    % scaling features by w2 is equivalent to a weighted Minkowski metric.
    w2        = sqrt(w(:)).';        % 1 x d
    xtrain_w  = xtrain .* w2;        % nTrain x d

    % -----------------------------
    % Allocate output
    % -----------------------------
    nK        = numel(k_values);
    predicted = zeros(nTest, nK);

    % -----------------------------
    % Main loop over test points
    % -----------------------------
    for i = 1:nTest

        xtest_w = xtest(i, :) .* w2;   % 1 x d : % scale test sample

        % Minkowski distances from current test point to all training points
        distances = pdist2(xtrain_w, xtest_w, 'minkowski', p);
        distances = distances(:);      % ensure column vector nTrain x 1

        % Sort distances (ascending)
        [sortedDist, indices] = sort(distances, 'ascend');

        % Loop over different k values
        for kk = 1:nK
            k = k_values(kk);

            neighbor_index = indices(1:k);
            d_k            = sortedDist(1:k);

            % Initialize weights for neighbors
            if fuzzy
                % Fuzzy kNN weights: w_i = d_i^(-1/(m-1))
                % (note: if distance is zero, weight becomes Inf --> we fix below)
                weight = d_k'.^(-1 / (m - 1));  % row vector 1 x k

                % Replace Inf (exact matches) with a large finite weight (e.g., max of others or 1)
                if any(isinf(weight))
                    % If all are Inf (all distances zero), use uniform weights
                    if all(isinf(weight))
                        weight = ones(size(weight));
                    else
                        weight(isinf(weight)) = max(weight(~isinf(weight)));
                    end
                end
            else
                % Classica k-NN: uniform weights
                weight = ones(1, k);
            end

            % Weighted average of target values
            % ytrain(neighbor_index,:) is k x r, weight is 1 x k
            % So: xtest_out = (weight * y_neighbors) / sum(weight)
            y_neighbors = ytrain(neighbor_index, :);   % k x r
            xtest_out   = (weight * y_neighbors) / sum(weight);

            predicted(i, kk) = xtest_out;
        end
    end

end