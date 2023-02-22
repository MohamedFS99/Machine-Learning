close all 
clc 
clear 
dir
% 1) Regulaized Linear Regression
% 1.1) Visualizing the dataset
% load X, Xtest, Xval, y, ytest, and yval from dataset
load('ex5data1.mat');
m = size(X,1); % m = number of examples

%plot training data
figure
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth',1.5);
xlabel('Change In Water Level (x)');
ylabel('Water Flowing Out Of The Dam (y)');

% 1.2) Regularized Linear Regression Cost Function
theta = [1; 1];
J = linearRegCostFunction([ones(m, 1) X], y, theta, 1);
fprintf("Cost Function at theta = [1; 1] = %f\n",J); %Ans = 303.993192

% 1.3) Regularized Linear Regression Gradient Descent
[J, grad] = linearRegCostFunction([ones(m,1) X], y, theta, 1);
fprintf("Graditent at theta [1;1] = [%f; %f]\n", grad(1), grad(2)) %Ans [-15.303016; 598.250744]

% 1.4) Fitting linear regression
% train linear regression with lambda = 0
lambda = 0;
[theta] = trainLinearReg(X, y, lambda);

% plot fit over the data
figure
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth',1.5);
xlabel("Change in Water Level (x)");
ylabel("Water Flowing Out of The Dam (y)");
hold on
plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth',2);
hold off

% 2) Bias-Variance
% 2.1) Learning Curves
lambda = 0;
[error_train, error_val] = learningCurve([ones(m,1) X], y, [ones(size(Xval,1),1) Xval], yval, lambda);

figure
plot(1:m, error_train, 1:m, error_val);
legend("Train", "Cross-Validation");
title("Learning Curves for Linear Regression");
xlabel("Number of Training Examples");
ylabel('Error');
axis([0 13 0 150]);

fprintf('# training Examples\tTrain Error\tCross Validation Error\n');
for i=1:m
    fprintf("   \t%d\t\t%f\t%f\n", i, error_train(i), error_val(i));
end

% 3) Ploynomial Regression
% 3.1) Learning Polynomial Regression

p = 8;
% Map X onto polynomial features and normalize
X_poly = polyFeatures(X,p);
[X_poly, mu, sigma] = featureNormalize(X_poly); %Normalize
X_poly = [ones(size(X_poly,1), 1), X_poly];                    % add ones

% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest,p);
X_poly_test = X_poly_test - mu;
X_poly_test = X_poly_test./sigma;
X_poly_test = [ones(size(X_poly_test,1) ,1), X_poly_test];

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval,p);
X_poly_val = X_poly_val - mu;
X_poly_val = X_poly_val./sigma;
X_poly_val = [ones(size(X_poly_val,1), 1), X_poly_val];

fprintf("Normalized Training Example 1: \n");
fprintf('   %f  \n', X_poly(1,:));


%Train Model
lambda = 0;
[theta] = trainLinearReg(X_poly, y, lambda);

% Plot training data and fit 
figure
plot(X,y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel("Change in Water Level (x)");
ylabel('Water Flowing Out of The Dam (y)');
title(sprintf('Polynomial Regression Fit (lambda = %f)', lambda));
[error_train, error_val] = learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val);
title(sprintf('Polynomial Regression Fit (lambda = %f)', lambda));
legend('Train', 'Cross Validation');
xlabel('Number of Training Examples');
ylabel('Error');
axis([0 13 0 100]);

% 3.2) Adjusting the regularization parameter
lambda = 1;
[theta] = trainLinearReg(X_poly, y, lambda);

% Plot training data and fit
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));
[error_train, error_val] = learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val);
title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')

% 3.3) Selecting lambda using a cross validation set
[lambda_vec, error_train, error_val] = validationCurve(X_poly, y, X_poly_val, yval);
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');
for i = 1:length(lambda_vec)
    if i == 1
        fprintf('lambda\t\tTrain Error\tValidation Error\n');
    end
    fprintf('%f\t%f\t%f\n',lambda_vec(i), error_train(i), error_val(i));
end

%3.5) Plotting learning curves with randomly selected examples
lambda = 0.01;
[error_train, error_val] = learningCurve(X_poly, y, X_poly_val, yval, lambda);

figure
plot(1:m, error_train, 1:m, error_val);
title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')
    