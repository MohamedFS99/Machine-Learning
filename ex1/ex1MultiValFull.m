clear

%load data into matlab
data = load('ex1data2.txt');
X = data(:, 1:2); %X1 is size in ft^2 and X2 is # of bedrooms
y = data(:,3); %y is price
m = length(y); %m is training example size

%show first 10 rows of data
% fprintf("X=[%.0f %.0f], y= %.0f\n", [X(1:10,:) y(1:10, :) ]);

%Apply feature normalization (feature scaling)
[X, mu, sigma] = featureNormalize(X);

%add bias (intercept) term to X
X = [ones(m,1) X];


% initialize alpha value
alpha = 0.01;
num_iters = 300;

% initialze theta
theta = zeros(3,1);
% run gradiant descend
[theta,J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
fprintf("J_history:\n\n", J_history(:))

%plot theta convergece
figure;
plot(1:numel(J_history), J_history, 'g', 'LineWidth',2);
xlabel("Number of Iterations")
ylabel("Cost of J")

% display results of gradiant descend
fprintf("The thetas computed from gradient descend for multiple" + ...
    " values: \n%.0f \n%.0f \n%.0f \n", theta(1), theta(2), theta(3));


%Estimate the price of a 1650 sq-ft, 3 br house using  gradient descend
house = [1650 3];
house_pred = (house - mu) ./ sigma;
house_pred_biased = [1, house_pred];
price1 = house_pred_biased * theta

% Solve with normal equations:
% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
norm_theta = normalEqn(X, y);

% Display normal equation's result
fprintf("The thetas computed from normal equation for multiple" + ...
    " values: \n%.0f \n%.0f \n%.0f \n", theta(1), theta(2), theta(3));

% Estimate the price of a 1650 sq-ft, 3 br house using normal equation
house2 = [1; 1650; 3];
price2 = norm_theta' * house2

