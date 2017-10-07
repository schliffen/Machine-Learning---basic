clear; clc; close all;

addpath('C:\Users\siahkoohi\Google Drive\Courses\CPSC 340\Homework\HW#2\340 HW (1)\a2');
addpath('C:\Users\siahkoohi\Google Drive\Courses\CPSC 340\Homework\HW#2\340 HW (1)');

%% Question 2
load('vowel.mat')

%% Question 2.1.1

train_error = zeros(15, 1);
test_error = zeros(15, 1);
for maxDepth = 1:15
    [model] = decisionTree(X,y,maxDepth);
    yhat_train = model.predict(model,X);
    train_error(maxDepth) = sum(y~=yhat_train)/length(y);
    yhat_test = model.predict(model,Xtest);
    test_error(maxDepth) = sum(ytest~=yhat_test)/length(ytest);
end

figure;
hold on
plot(1:15, train_error, '--gs', 'LineWidth', 2, 'MarkerSize',5,...
    'MarkerEdgeColor','b', 'MarkerFaceColor',[0.5,0.5,0.5])
plot(1:15, test_error, '--rs', 'LineWidth', 2, 'MarkerSize',5,...
    'MarkerEdgeColor','b', 'MarkerFaceColor',[0.5,0.5,0.5])
title('(Question 2.1.1) Testing and training error rate')
xlabel('Depth')
ylabel('Training Error')
legend('Training error','Testing error','Location','northeast')
axis([1, 15, min(min(train_error), min(test_error)) ...
    , max(max(test_error), max(train_error))]);


%% Question 2.1.3

train_error = zeros(15, 1);
test_error = zeros(15, 1);
for maxDepth = 1:15
    [model] = randomTree(X,y,maxDepth);
    yhat_train = model.predict(model,X);
    train_error(maxDepth) = sum(y~=yhat_train)/length(y);
    yhat_test = model.predict(model,Xtest);
    test_error(maxDepth) = sum(ytest~=yhat_test)/length(ytest);
end

figure;
hold on
plot(1:15, train_error, '--gs', 'LineWidth', 2, 'MarkerSize',5,...
    'MarkerEdgeColor','b', 'MarkerFaceColor',[0.5,0.5,0.5])
plot(1:15, test_error, '--rs', 'LineWidth', 2, 'MarkerSize',5,...
    'MarkerEdgeColor','b', 'MarkerFaceColor',[0.5,0.5,0.5])
title('(Question 2.1.3) Testing and training error rate')
xlabel('Depth')
ylabel('Training Error')
legend('Training error','Testing error','Location','northeast')
axis([1, 15, min(min(train_error), min(test_error)) ...
    , max(max(test_error), max(train_error))]);


%% Question 2.1.4

train_error = zeros(15, 1);
test_error = zeros(15, 1);
for maxDepth = 1:15
    [model] = decisionTree_rand(X,y,maxDepth);
    yhat_train = model.predict(model,X);
    train_error(maxDepth) = sum(y~=yhat_train)/length(y); 
    yhat_test = model.predict(model,Xtest);
    test_error(maxDepth) = sum(ytest~=yhat_test)/length(ytest);
end

figure;
hold on
plot(1:15, train_error, '--gs', 'LineWidth', 2, 'MarkerSize',5,...
    'MarkerEdgeColor','b', 'MarkerFaceColor',[0.5,0.5,0.5])
plot(1:15, test_error, '--rs', 'LineWidth', 2, 'MarkerSize',5,...
    'MarkerEdgeColor','b', 'MarkerFaceColor',[0.5,0.5,0.5])
title('(Question 2.1.4) Testing and training error rate')
xlabel('Depth')
ylabel('Training Error')
legend('Training error','Testing error','Location','northeast')
axis([1, 15, min(min(train_error), min(test_error)) ...
    , max(max(test_error), max(train_error))]);

%% Question 2.2.1 

depth = inf;
nBootstraps = 50;

[model] = decisionForest(X,y,depth,nBootstraps);
yhat_test = model.predict(model,Xtest);
test_error = sum(ytest~=yhat_test)/length(ytest);

fprintf('(Question 2.2.1) Testing error is: %g \n',...
    test_error)

%% Question 2.2.2

[model] = decisionForest_rand(X,y,depth,nBootstraps);
yhat_test = model.predict(model,Xtest);
test_error = sum(ytest~=yhat_test)/length(ytest);

fprintf('(Question 2.2.2) Testing error is: %g \n',...
    test_error)

%% Question 2.2.3

[model] = decisionForest_randomTree(X,y,depth,nBootstraps);
yhat_test = model.predict(model,Xtest);
test_error = sum(ytest~=yhat_test)/length(ytest);

fprintf('(Question 2.2.3) Testing error is: %g \n',...
    test_error)

%% Question 2.2.4

[model] = decisionForest_randomTree_rand(X,y,depth,nBootstraps);
yhat_test = model.predict(model,Xtest);
test_error = sum(ytest~=yhat_test)/length(ytest);

fprintf('(Question 2.2.4) Testing error is: %g \n',...
    test_error)










