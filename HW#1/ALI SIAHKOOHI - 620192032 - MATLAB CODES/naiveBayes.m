function [model] = naiveBayes(X,y)
% [model] = naiveBayes(X,y,k)
%
% Implementation of navie Bayes classifier for binary features

% Compute number of training examples and number of features
[n,d] = size(X);

% Computer number of class lables
k = max(y);

counts = zeros(k,1);
for c = 1:k
    counts(c) = sum(y==c);
end
p_y = counts/n; % This is the probability of each class, p(y(i) = c)

% We will store:
%   p(x(i,j) = 1 | y(i) = c) as p_xy(j,1,c)
%   p(x(i,j) = 0 | y(i) = c) as p_xy(j,2,c)
for c = 1:k
    ind = find(c==y);
    for j = 1:d
        tmp = sum(1==X(ind, j));
        a = zeros(1); a(1) = tmp(1);
        p_xy(j, 1, c) = a/length(ind);
    end
end
p_xy(:, 2, :) = 1 - p_xy(:, 1, :);


model.k = k;
model.p_y = p_y;
model.p_xy = p_xy;
model.predict = @predict;
end

function [yhat] = predict(model,Xtest)
[t,d] = size(Xtest);
k = model.k;
p_y = model.p_y;
p_xy = model.p_xy;

yhat = zeros(t,1);
for i = 1:t
    probs = p_y; % This will be the probability for each class
    for j = 1:d
        if Xtest(i,j) == 1
            for c = 1:k
                probs(c) = probs(c)*p_xy(j,1,c);
            end
        else
            for c = 1:k
                probs(c) = probs(c)*p_xy(j,2,c);
            end
        end
    end
    [maxProb,yhat(i)] = max(probs);
end
end