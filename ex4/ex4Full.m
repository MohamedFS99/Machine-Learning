clear
close all
clc
dir


% 1) Neural Networks
% 1.1) visualizing the data

load("ex4data1.mat");
m = size(X, 1);

% randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);
displayData(X(sel, :));

% 1.2) model representation
load("ex4weights.mat");

% 1.3) feed forward and cost function
input_layer_size = 400; % 20 x 20 matrix
hidden_layer_size = 25; % 25 hidden layers
num_labels = 10; % 10 labels from 1 to 10 ("0" was labeled as label 10

% unroll parameters
nn_parameters = [Theta1(:) ; Theta2(:)];

% weight regularization parameter (no regularization now so set to 0)
lambda = 0;

% apply cost function (no regularization)
J = nnCostFunction(nn_parameters,input_layer_size, hidden_layer_size, num_labels, X,y, lambda);
fprintf('Cost at parameters (loaded from ex4weights.mat = %f\n', J);

% 1.4) regularized cost function
% weight regularization parameter
lambda = 1;

% apply cost function (with regularization)
J = nnCostFunction(nn_parameters, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
fprintf('Cost at parameters(loaded from ex4weights.mat = %f \n', J)

% 2) Backpropagation
% 2.1) sigmoid gradient
sigmoidGradient(0)

% 2.2) random initialization
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% unroll parameters
initial_nn_parameters = [initial_Theta1(:); initial_Theta2(:)];

% good way to choose epsilon_init is sqrt(6) / sqrt(L_in + L_out) as L_in
% is s_(l) and L_out is s_(l+1)  are the number of units in the layers 
% adjacent to Thetal(l <- L small).

% 2.3) backpropagation
% implemented in nnCostFunction

% 2.4) gradient checking
% checkNNGradients;
% Relative Difference: 2.33553e-11 < 1e-9

% 2.5) regularized neural network
% check gradients by running checkNNGradients.m
lambda = 3;
% checkNNGradients(lambda); % Relative Difference: 2.25401e-11

% output the costFunction debugging value (0.576051)
debug_J = nnCostFunction(nn_parameters, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
fprintf('Cost at (fixed) debugging parameters (W/ lambda = 3) = %f\n', debug_J); 

% 2.6) learning parameters using fmincg
options = optimset('MaxIter', 50);
lambda = 1;

% create a short hand for the cost function
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels,X,y,lambda);

% the cost function now takes only the nn parameters as an input parameter
[nn_parameters, ~] = fmincg(costFunction, initial_nn_parameters, options);

% obtain Theta1 and Theta2 from nn parameters
Theta1 = reshape(nn_parameters(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_parameters(1 + hidden_layer_size * (input_layer_size + 1): end), num_labels, (hidden_layer_size + 1));

% predict and find training set accuracy
pred = predict(Theta1, Theta2, X);
fprintf('Training set accuracy = %f\n', mean(double(pred == y)) * 100); % accuracy = 95.3% +- 1.0%

% 3) visualize the hidden layer
% visualize weights
displayData(Theta1(:, 2:end));