close all 
clear
dir

%load data
data = load("ex2data1.txt");
X = data(:, [1,2]);
y = data(:, 3);

%1.1)plot examples data using + and o indicators
plotData(X,y);

%labels
xlabel("Exam 1 Grades");
ylabel("Exam 2 Grades");
legend("Admitted", "Not Admitted");

% 1.2) Implementation
%1.2.1) sigmoid function
sigmoid(0);

%1.2.2) cost function

% initialize data
%setup data matrix
[m,n] = size(X);
% create intercept of X
X = [ones(m,1) X];
%initialize fitting parameters (theta)
initial_theta = zeros(n+1, 1);

% compute cost and gradient
%cost and gradient of initial theta(zeros)
[cost,grad] = costFunction(initial_theta, X, y);
fprintf("\nCost for initial theta (zeros): %f\n", cost);
disp("Gradient for initial theta (zeros):"); disp(grad);
% cost and gradient non-zero theta (test theta)
test_theta = [-24; 0.2; 0.2];
[cost, grad] = costFunction(test_theta, X, y);
fprintf("\nCost for non-zero test theta: %f\n", cost);
disp("Gradient for non-zero test theta:"); disp(grad);

%1.2.3) learning parametes using fminunc
options = optimoptions(@fminunc, 'Algorithm', 'Quasi-Newton', 'GradObj', 'on', 'MaxIter', 400);
%run fminunc to compute cost and optimal theta
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

%print cost and theta
fprintf("Cost at theta found by fminunc:%f\n",cost);
disp("theta: "); disp(theta);

%plot decision boundary
plotDecisionBoundary(theta, X, y);
%add labels and legend
hold on
xlabel('Exam 1 Grades');
ylabel("Exam 2 Grades");
legend('Admitted', 'Not Admitted');
hold off

% 1.2.4) Evaluating logistic regression
% predict probability of student getting 45 in exam 1 and 85 in exam 2
prob = sigmoid([1 45 85] *theta);
fprintf("A student with the exam grades of 45 and 85 has the probability of admission predicted to: %f\n\n", prob);
% compute training set accuracy
p = predict(theta, X);
fprintf("Accuracy of logistic regression training model is: %f\n ", mean(double(p==y) * 100))

% 2) Regularized logistic regression
% 2.1) visualize data
theta(1) = 0;data = load('ex2data2.txt');
X = data(:, [1, 2]);
y = data(:, 3);

% plot data
plotData(X, y);

% add labels and legend
hold on
xlabel('Microship Test 1');
ylabel('Microship Test 2');
legend('y=1', 'y=0');
hold off

% 2.2) Feature Mapping
% add polynoyal features using mapFeature function
X = mapFeature(X(:,1), X(:, 2));

% 2.3) Regularized Cost Function and Gradient
% initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% set regularizing paramather (lambda) to 1
lambda = 1;

% Compute and display regularized cost function and gradient descent for logistic regression

% initial (zero) theta case
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
fprintf("Cost at initial theta (Zeros) is: %f\n", cost);
fprintf("First 5 values of Gradient Descent for initial theta (zeros) is: \n") ;
fprintf("%f \n", grad(1:5));

% test (ones) theta case and lambda = 10
test_theta = ones(size(X, 2), 1);
[cost, grad] = costFunctionReg(test_theta, X, y, 10);
fprintf("\nCost Function for test theta (ones) is: %f\n", cost);
fprintf("First 5 values of Gradient Descent for test theta (ones) is: \n");
fprintf("%f \n", grad(1:5));

% 2.3) learning parametes using fminunc
options = optimoptions(@fminunc, 'Algorithm', 'Quasi-Newton', 'GradObj','on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t)(costFunctionReg(t, X, y, lambda)),initial_theta, options);


% print cost and theta
fprintf("Regualrized Cost at theta found by fminunc:%f\n",cost);
disp("regualrized theta: "); disp(theta);

% 2.4) plot decision boundary
plotDecisionBoundary(theta, X, y);
hold on
xlabel('Microship Test 1');
ylabel('Microship Test 2');
legend('y=1', 'y=0', 'decision boundary');
hold off
