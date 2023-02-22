clear
close all
clc
dir

% 1) Multi Class Classification

% 1.1) Dataset
% Load dataset
load('ex3data1.mat');

% 1.2) Data Visualization
m = size(X,1);
% randomly select 100 data points to visalize
rand_indencies = randperm(m);
sel = X(rand_indencies(1:100), :);
% display data
displayData(sel);

% 1.3) Vectorizing logistic regression
theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15, 5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J, grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);
fprintf("Cost: %f | Expected cost: 2.534819\n",J);
fprintf("Gradients: \n");fprintf(" %f \n", grad);
fprintf('Expected gradients:\n 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

% 1.4) One vs All Classification
a = 1:10;
b = 6;
disp(a == b);

num_labels = 10; % 10 labels from 1 to 10
lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

% 1.4.1) One vs All prediction
pred = predictOneVsAll(all_theta, X);
fprintf("\nTraining Set Accuracy (One vs All Classificaton): %f\n", mean(double(pred == y)) * 100);

% 2) Neural Networks
% 2.1) Model representation
load ("ex3data1.mat");
m = size(X, 1);

% randomly select 100 data inputs to display
sel = randperm(size(X, 1));
sel = sel(1:100);
displayData(X(sel, :));

% load saved matrices from file
load("ex3weights.mat");
% Theta1 size 25 x 401
% Theta2 size 10 x 26

% 2.2) Feedforward propagation and prediction
pred = predict(Theta1, Theta2, X);
fprintf("\nTraining Set Accuracy (Neural Network): %f\n", mean(double(pred == y)) *100);

% randomly permute examples
rp = randi(m);
%predict
pred = predict(Theta1, Theta2, X(rp, :));
fprintf("\nNeural Network prediction: %d (digit %d) \n", pred, mod(pred,10));
%display data
displayData(X(rp, :));
