clear
dir

% 1. Support Vector Machines
% 1.1 example dataset 1

load('ex6data1.mat');

plotData(X,y);

C = 1; % 1, 10, 50, 100
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);


% 1.2 SVM with gaussian kernels
% 1.2.1 Gaussian kernel

x1 = [1 2 1]; x2 = [0 4 -1]; 
sigma = 2; % 2, 0.5, 4, 0.25, 8, 16
sim = gaussianKernel(x1,x2,sigma);
fprintf("The Gaussian Kernel between x1 = [1 2 1] , x2 = [0 4 -1], sigma = %f : \n \t%g\n", sigma, sim);

% 1.2.2 Example dataset 2

load('ex6data2.mat');

plotData(X,y);

% SVM parameters
C = 1; sigma = 0.1;

% We set the tolerance and max_passes lower here so that the code will run faster. However, in practice, 
% you will want to run the training to convergence.

model = svmTrain(X, y, C, @(x1,x2) gaussianKernel(x1,x2, sigma));
visualizeBoundary(X, y, model);


% 1.2.3 Example dataset 3

load('ex6data3.mat');

plotData(X,y);

% Try different SVM Parameters here
[C, sigma] = dataset3Params(X, y, Xval, yval);

% Train the SVM
model = svmTrain(X, y, C, @(x1, x2)gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);

% 2. Spam Classification

% 2.1 Preprocessing emails

% 2.1.1 Vocabulary list

% Initialization
clear;

% extrach features
file_contents = readFile('emailSample1.txt');
word_indices = processEmail(file_contents);

% display stats
disp(word_indices)


% 2.2 Extracting features from emails

% Extract Features
features = emailFeatures(word_indices);

% Print Stats
fprintf('Length of feature vector: %d\n', length(features));
fprintf('Number of non-zero entries: %d\n', sum(features > 0));


% 2.3 Training SVM for spam classification

% Load the spam email dataset
% You will have X and y in environment
load('spamTrain.mat');

C = 0.1;
model = svmTrain(X, y, C, @linearKernel);

p = svmPredict(model, X);
fprintf("Training Accuracy = %f\n", mean(double(p == y)) * 100);

% load test dataset
% You will have Xtest, ytest in your environment
load('spamTest.mat');

p = svmPredict(model, Xtest);
fprintf("Test Accuracy = %f\n", mean(double( p == ytest)) * 100);


% sort weights and obtin the vocabulary list
[weight, index] = sort(model.w, 'descend');
vocabList = getVocabList();
for i = 1:15
    if i == 1
        fprintf("Top predictors of Spam \n");
    end
    fprintf('%-15s (%f) \n', vocabList{index(i)}, weight(i));
end




