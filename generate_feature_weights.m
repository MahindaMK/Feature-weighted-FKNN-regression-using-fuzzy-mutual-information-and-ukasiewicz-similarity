function [wRel,wRed,wDep,wSum] = generate_feature_weights(data,type)

% GENERATE_FEATURE_WEIGHTS
%   Compute feature weights based on relevance, redundancy, and dependency using Łukasiewicz-based fuzzy similarity.
%
%   [wRel, wRed, wDep, wSum] = generate_feature_weights(data, data_ts, type, setSize)
%
%   Inputs:
%       data    : (nSamples x (nFeatures+1)) matrix.
%                 Last column       = class labels or target.
%       type    : 1  -> classification (uses MC on class labels)
%                 0  -> regression    (uses simR on target values)
%
%   Outputs:
%       wRel : 1 x nFeatures vector: Relevancy weights with Lukasiwicz based similarity
%       wRed : 1 x nFeatures vector: Redundancy weight with Lukasiwicz based similarity
%       wDep : 1 x nFeatures vector: Depencendy weights with Lukasiwicz based similarity
%       wSum : 1 x nFeatures vector: Sum based combination of them
%
%   Note:
%       This function assumes functions MC, simR, simL, FHjoint,
%       and dependencycomp are available on the MATLAB path.
%
%   Created by Mahinda Mailagaha Kumbure & Pasi Luukka

    % -----------------------------
    % Basic setup
    % -----------------------------
    p = 1;  % parameter for Łukasiewicz-based similarity (simL)

    % Separate features and class/target
    y = data(:, end);        % class labels or regression target
    X = data(:, 1:end-1);    % features
    [~, nFeatures] = size(X);

    % -----------------------------
    % Class / target variable
    % -----------------------------
    if type == 1
        % Classification case
        RC = MC(y);
    else
        % Regression case
        RC = simR(y, y);
    end

    % -----------------------------
    % Precompute feature similarity relations
    % -----------------------------
    Rfeat = cell(1, nFeatures);
    for f = 1:nFeatures
        Rfeat{f} = simL(X(:, f), X(:, f), p);
    end

    % -----------------------------
    % 1. Relevance weights
    % -----------------------------
    Rel = zeros(1, nFeatures);
    for f = 1:nFeatures
        Rel(f) = FHjoint(Rfeat{f}, RC);
    end
    wRel = Rel / sum(Rel);  % normalize relevance weights

    % -----------------------------
    % 2. Redundancy weights
    % -----------------------------
    Redundancy = zeros(1, nFeatures);
    for i = 1:nFeatures
        Ri = Rfeat{i};
        redTmp = zeros(1, nFeatures);
        for j = 1:nFeatures
            Rj = Rfeat{j};
            redTmp(j) = FHjoint(Ri, Rj);
        end
        Redundancy(i) = sum(redTmp);
    end

    % Lower redundancy --> higher usefulness --> invert and normalize
    invRed = 1 ./ Redundancy;
    wRed   = invRed / sum(invRed);

    % -----------------------------
    % 3. Dependency weights
    % -----------------------------
    degree = zeros(nFeatures, nFeatures);
    for i = 1:nFeatures
        Ri = Rfeat{i};
        for j = 1:nFeatures
            Rj = Rfeat{j};
            degree(i, j) = dependencycomp(Ri, Rj, RC);
        end
    end

    depDeg = sum(degree, 1);     % aggregate per feature
    wDep   = depDeg / sum(depDeg);

    % -----------------------------
    % 4. Combined weights
    % -----------------------------
    wTmp = wRel + wRed + wDep;
    wSum = wTmp / sum(wTmp);





   




