%% Animals with attributes data
load animals.mat

%% DBSCAN clustering
radius = 13;
minPts = 3;
doPlot = 0;

model = clusterDBcluster(X,radius,minPts,doPlot);
k = max(model.y) - min(model.y)
for c = 1:k
    fprintf('Cluster %d: ',c);
    fprintf('%s ',animals{model.y==c});
    fprintf('\n');
end


% radius = 500;
% minPts = 6;