function [model] = clusterKmedians(X,k,doPlot)
% [model] = clusterKmedians(X,k,doPlot)
%
% K-medians clustering

[n,d] = size(X);
y = ones(n,1);

% Choose random points to initialize medians
W = zeros(k,d);
for k = 1:k
    i = ceil(rand*n);
    W(k,:) = X(i,:);
end

X2 = X.^2*ones(d,k);
while 1
    y_old = y;
    
    % Draw visualization
    if doPlot && d == 2
        clustering2Dplot(X,y,W)
    end
    
    % Compute (absolute) Euclidean distance between each data point and
    % each median
    distances = zeros(size(X, 1), size(W, 1));
    for i = 1:size(distances, 1)
        for j = 1:size(distances, 2)
            distances(i, j) = norm(X(i, :) - W(j, :), 1);
        end
    end
    
    % Assign each data point to closest mmedian
    [~,y] = min(distances,[],2);
    
    % Draw visualization
    if doPlot && d == 2
        clustering2Dplot(X,y,W)
    end
    
    % Compute median of each cluster
    for k = 1:k
        W(k,:) = median(X(y==k,:),1);
    end
    
    changes = sum(y ~= y_old);
    fprintf('Running K-medians, difference = %f\n',changes);
    
    % Stop if no point changed cluster
    if changes == 0
        break;
    end
end

model.W = W;
model.y = y;
model.predict = @predict;
model.error = @error;
end

function [y] = predict(model,X)
[t,d] = size(X);
W = model.W;
k = size(W,1);

% Compute Euclidean distance between each data point and each median
distances = zeros(size(X, 1), size(W, 1));
for i = 1:size(distances, 1)
    for j = 1:size(distances, 2)
        distances(i, j) = norm(X(i, :) - W(j, :), 1);
    end
end


% Assign each data point to closest mean
[~,y] = min(distances,[],2);
end

function [error] = error(model,X)

[n,d] = size(X);
W = model.W;
k = size(W, 1);
X2 = X.^2*ones(d,k);

distances = zeros(size(X, 1), size(W, 1));
for i = 1:size(distances, 1)
    for j = 1:size(distances, 2)
        distances(i, j) = norm(X(i, :) - W(j, :), 1);
    end
end

error = 0;

for i = 1:500
    error = error + (min(distances(i, :), [], 2));
end

end



