clear; clc; close all;

addpath('C:\Users\siahkoohi\Google Drive\Courses\CPSC 340\Homework\HW#2\340 HW (1)\a2');
addpath('C:\Users\siahkoohi\Google Drive\Courses\CPSC 340\Homework\HW#2\340 HW (1)');

%% Question 1.1
%% Question 1.1.2 (k = 1)

load('citiesSmall.mat');

k = 1;
model = knn(X,y,k);
yhat_train = model.predict(model,X);
error_train = sum(y~=yhat_train)/length(y);
fprintf('(Question 1.1.2) Training error for (k = 1) is: %g \n',...
    error_train)

yhat_test = model.predict(model,Xtest);
error_test = sum(ytest~=yhat_test)/length(ytest);
fprintf('(Question 1.1.2) Testing error for (k = 1) is: %g \n',...
    error_test)

classifier2Dplot(X,y,Xtest,ytest,model)
title('(Question 1.1.3) Classifier 2D plot for k = 1')

%% Question 1.1.2 (k = 3)

k = 3;
model = knn(X,y,k);
yhat_train = model.predict(model,X);
error_train = sum(y~=yhat_train)/length(y);
fprintf('(Question 1.1.2) Training error for (k = 3) is: %g \n',...
    error_train)

yhat_test = model.predict(model,Xtest);
error_test = sum(ytest~=yhat_test)/length(ytest);
fprintf('(Question 1.1.2) Testing error for (k = 3) is: %g \n',...
    error_test)

%% Question 1.1.2 (k = 10)

k = 10;
model = knn(X,y,k);
yhat_train = model.predict(model,X);
error_train = sum(y~=yhat_train)/length(y);
fprintf('(Question 1.1.2) Training error for (k = 10) is: %g \n',...
    error_train)

yhat_test = model.predict(model,Xtest);
error_test = sum(ytest~=yhat_test)/length(ytest);
fprintf('(Question 1.1.2) Testing error for (k = 10) is: %g \n',...
    error_test)


%% Question 1.2
%% Question 1.2.2

load('citiesBig1.mat');
k = 1;

model = cnn(X,y,k);
fprintf('(Question 1.2.2) Number of variables in the subset (k = 1) is: %g \n',...
    length(model.y))

yhat_train = model.predict(model,X);
error_train = sum(y~=yhat_train)/length(y);
fprintf('(Question 1.2.2) Training error for (k = 1) is: %g \n',...
    error_train)

yhat_test = model.predict(model,Xtest);
error_test = sum(ytest~=yhat_test)/length(ytest);
fprintf('(Question 1.2.2) Testing error for (k = 1) is: %g \n',...
    error_test)


%% Question 1.2.3

figure;
classifier2Dplot(X,y,Xtest,ytest,model)
title('(Question 1.2.3) Classifier 2D plot for k = 1')

%% Question 1.2.6

load('citiesBig2.mat');
k = 1;

model = cnn(X,y,k);
fprintf('(Question 1.2.6) Number of variables in the subset (k = 1) is: %g \n',...
    length(model.y));

yhat_train = model.predict(model,X);
error_train = sum(y~=yhat_train)/length(y);
fprintf('(Question 1.2.6) Training error for (k = 1) is: %g \n',...
    error_train)

yhat_test = model.predict(model,Xtest);
error_test = sum(ytest~=yhat_test)/length(ytest);
fprintf('(Question 1.2.6) Testing error for (k = 1) is: %g \n',...
    error_test)

p = randperm(length(y))';
X_perm = X(p, :);
y_perm = y(p);

model = cnn(X_perm,y_perm,k);
fprintf('(Question 1.2.6) Number of variables in the subset (k = 1) is: %g \n',...
    length(model.y));

yhat_train = model.predict(model,X_perm);
error_train = sum(y_perm~=yhat_train)/length(y_perm);
fprintf('(Question 1.2.6) Training error for (k = 1) is: %g \n',...
    error_train)

yhat_test = model.predict(model,Xtest);
error_test = sum(ytest~=yhat_test)/length(ytest);
fprintf('(Question 1.2.6) Testing error for (k = 1) is: %g \n',...
    error_test)
