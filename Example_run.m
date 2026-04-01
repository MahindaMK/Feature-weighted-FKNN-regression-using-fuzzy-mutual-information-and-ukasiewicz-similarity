%% Example: Using FWMD-FKNNreg on the Anacalt Dataset
% Created by: Mahinda Mailagaha Kumbure & Pasi Luukka
% Date:       03/2026
% ==============================================================

clear; clc; close all;

% Load the data (example data: Anacalt)
dataTbl = readtable('data_ANACALT.dat');  % from UCI
data    = table2array(dataTbl);          % numeric matrix

% Normalize all columns to [0,1]
dataNorm = normalize(data, 'range');

% Separate features and target
X = dataNorm(:, 1:end-1);   % features
Y = dataNorm(:, end);       % target
nFeatures = size(X, 2);

% Training and validation

% Parameter initialization
m       = 1.5;       % fuzzifier parameter (>1)
k       = 5;         % number of nearest neighbours
p       = 1.5;       % Minkowski distance order
valFrac = 0.5;       % fraction for hold-out validation (50/50)

% Hold-out cross-validation split
cv      = cvpartition(size(X, 1), 'HoldOut', valFrac);
idxTest = cv.test;

Xtrain  = X(~idxTest, :); % train data with n samples and m features
Ytrain  = Y(~idxTest, :); % target values of train samples 
Xtest   = X(idxTest, :); % test data with D samples and m features
Ytest   = Y(idxTest, :); % target values of test samples    

% Generate feature weights (relevance, redundancy, dependency)

% Type = 1 (classification), Type = 2 (regression)
[wRel, wRed, wDep, wSum] = generate_feature_weights([Xtrain, Ytrain], 2);

% We can switch between weight types easily:
% selected_weights = wRel;   % relevance-based
% selected_weights = wRed;   % redundancy-based 
% selected_weights = wDep;   % dependency-based

selected_weights = wRel;     % example: use relevance-based weights

% Run FWMD-FKNNreg (weighted fuzzy k-NN regression)

useFuzzy   = true;  % fuzzy regression
predicted_fknnw = mink_weighted_fknnreg(Xtrain, Ytrain, Xtest, k, useFuzzy, m, p, selected_weights);

% Evaluate performance (RMSE and R2)

rmse_fknnw = sqrt(mean((Ytest - predicted_fknnw).^2));
R2_fknnw   = r2(Ytest, predicted_fknnw);

fprintf('FWMD-FKNNreg (relevance-based weights): RMSE = %.4f, R^2 = %.4f\n', rmse_fknnw, R2_fknnw);
    


