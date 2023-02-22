clear

%run warm up exercise function
warmUpExercise()

% load data into matlab
data = load('ex1data1.txt');
X = data(:,1); y = data(:,2); % X is first column and y is second column

%plot using plotData function
plotData(X,y);

% implementing gradient descend
m = length(X); %training examples size
X = [ones(m,1), data(:,1)]; %add column of 1 as x0 
theta = zeros(2,1); 
iterations = 1500;
alpha = 0.01;

% Compute and display initial cost with theta all zeros
J_zero = computeCost(X, y, theta);

% Compute and display initial cost with non-zero theta
J_non_zero = computeCost(X, y,[-1; 2]);

% run gradient descend
theta = gradientDescent(X, y, theta, alpha, iterations);

% print theta to screen
fprintf("The theta computed from gradient descend: \n%f \n%f", theta(1), theta(2));

%plot the linear regression fit
hold on %hold old figure
plot(X(:,2), X*theta, '-')
legend("Training Data", "Linear Regression")
hold off

%predict using gradient descent
% population 1 to predict for 35,000. X axis in data is per 10,000 so
% 35,000/10,000 = 3.5
population1= [1, 3.5];
prediction1 = population1 * theta;
fprintf("\nFor the population of 35,000, we predict the revenue of %f\n", prediction1*10000);
%multiplied by 10,000 because data is per 10,000

%prediction 2 is same as prediction 1 but for 70,000
population2 = [1, 7];
prediction2 = population2 * theta;
fprintf("For the population of 70,000, we predict the revenue of %f\n", prediction2*10000);

% visualizing J(theta0, theta1)
theta0_values = linspace(-10,10,100);
theta1_values = linspace(-1,4,100);

J_values = zeros(length(theta0_values), length(theta1_values));

for i=1:length(theta0_values)
    for j=1:length(theta1_values)
        t = [theta0_values(i); theta1_values(j)];
        J_values(i,j) = computeCost(X,y,t);
    end
end

% plot J(theta0, theta1)
J_values = J_values'; %surf command will flip the J values so we need to transpose it
figure;
surf(theta0_values, theta1_values, J_values);
xlabel("\theta_0"); ylabel("\theta_1")
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_values, theta1_values, J_values, logspace(-2,3,20)); 
hold on
xlabel("\theta_0"); ylabel("\theta_1");
plot(theta(1), theta(2), 'rx', 'MarkerSize',10, "LineWidth",2);
hold off
