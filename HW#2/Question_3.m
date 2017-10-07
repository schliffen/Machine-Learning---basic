clear; clc; close all;

addpath('/Users/alisk/Desktop/340 HW/a2');
addpath('C:\Users\siahkoohi\Google Drive\Courses\CPSC 340\Homework\HW#2\340 HW\340 HW\a2');

%% Question 3
load ('clusterData.mat')

%% Question 3.1.2
doPlot = 0;
k = 4;

mine = 1e10;
for i = 1:50
    model = clusterKmeans(X, k, doPlot);
    error = model.error(model,X);
    
    if error < mine
        minW = model.W;
        mine = error;
        Model = model;
    end
end

figure
clustering2Dplot(X,Model.y,minW)
title('(Question 3.1.2)')

%% Question 3.2.3
% doPlot = 0;
K = 10;
mine = zeros(10, 1);
for k = 1:K
    
    mine(k) = 1e10;
    for i = 1:50
        model = clusterKmeans(X, k, doPlot);
        error = model.error(model,X);
        
        if error < mine(k)
%            minW = model.W;
           mine(k) = error;
           Model = model;
        end
    end
end


figure;
hold on
plot(1:K, mine, '--gs', 'LineWidth', 2, 'MarkerSize',5,...
    'MarkerEdgeColor','b', 'MarkerFaceColor',[0.5,0.5,0.5])
title('(Question 3.2.3) Choosing k!, the minimun error for different values of k')
xlabel('k')
ylabel('minimum error')

%% Question 3.3.1
clear;
load ('clusterData2.mat')
doPlot = 0;
k = 4;

mine = 1e10;
for i = 1:50
    model = clusterKmeans(X, k, doPlot);
    error = model.error(model,X);
    
    if error < mine
        minW = model.W;
        mine = error;
        Model = model;
    end
end

figure
clustering2Dplot(X,Model.y,minW)
title('(Question 3.3.1) Choosing k!, the minimun error for different values of k')
%% Question 3.3.2

doPlot = 0;
K = 10;
mine = zeros(10, 1);
for k = 1:K
    
    mine(k) = 1e10;
    for i = 1:50
        model = clusterKmeans(X, k, doPlot);
        error = model.error(model,X);
        
        if error < mine(k)
%            minW = model.W;
           mine(k) = error;
           Model = model;
        end
    end
end


figure;
hold on
plot(1:K, mine, '--gs', 'LineWidth', 2, 'MarkerSize',5,...
    'MarkerEdgeColor','b', 'MarkerFaceColor',[0.5,0.5,0.5])
title('(Question 3.3.2) Choosing k!, the minimun error for different values of k')
xlabel('k')
ylabel('minimum error')


%% Question 3.3.4

doPlot = 0;
K = 10;
mine = zeros(10, 1);
for k = 1:K
    
    mine(k) = 1e10;
    for i = 1:50
        model = clusterKmedians(X, k, doPlot);
        error = model.error(model,X);
        
        if error < mine(k)
%            minW = model.W;
           mine(k) = error;
           Model = model;
        end
    end
end


figure;
hold on
plot(1:K, mine, '--gs', 'LineWidth', 2, 'MarkerSize',5,...
    'MarkerEdgeColor','b', 'MarkerFaceColor',[0.5,0.5,0.5])
title('(Question 3.3.4) Choosing k!, the minimun error for different values of k')
xlabel('k')
ylabel('minimum error')



k = 4;

mine = 1e10;
for i = 1:50
    model = clusterKmedians(X, k, doPlot);
    error = model.error(model,X);
    
    if error < mine
        minW = model.W;
        mine = error;
        Model = model;
    end
end

figure
clustering2Dplot(X,Model.y,minW)