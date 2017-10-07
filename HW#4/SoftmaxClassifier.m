function [model] = SoftmaxClassifier(X,y)
% Classification using one-vs-all least squares

% Compute sizes
[n,d] = size(X);
k = max(y);
maxFunEvals = 400; % Maximum number of evaluations of objective
verbose = 0; % Whether or not to display progress of algorithm

W = zeros(d,k); % Each column is a classifier
for c = 1:k
%     yc = ones(n,1); % Treat class 'c' as (+1)
%     yc(y ~= c) = -1; % Treat other classes as (-1)
    W = findMin(@SoftmaxLoss,zeros(d*k,1),maxFunEvals,verbose,X,y);
end
W = reshape(W,[d k]);
model.W = W;
model.predict = @predict;
end

function [yhat] = predict(model,X)
W = model.W;
[~, yhat] = max(X*W,[], 2);
end

function [f,G] = SoftmaxLoss(W,X,y)
W = reshape(W,[size(X, 2) max(y)]);
tmp_tot = 0; 
tmp = 0;
for i = 1:size(X,1)
    tmp_tot = tmp_tot - W(:, y(i))'*X(i, :)';
    tmp = 0;
    for cp = 1:max(y)
        tmp = tmp + exp(W(:, cp)'*X(i, :)');
    end
    tmp = log(tmp);
    tmp_tot = tmp_tot + tmp;
end

f = tmp_tot;

tmp_tot = 0; 
tmp = 0;
for j = 1:size(X, 2)
    for c = 1:max(y);
        for i = 1:size(X,1)
            tmp_tot = tmp_tot - X(i, j)*(y(i)==c);
            tmp = 0;
            for cp = 1:max(y)
                tmp = tmp + exp(W(:, cp)'*X(i, :)');
            end
            tmp = 1./(tmp)*X(i, j)*exp(W(:, c)'*X(i, :)');
            tmp_tot = tmp_tot + tmp;
        end
        G(j, c) = tmp_tot;
        tmp_tot = 0;
    end
end
G = G(:);


end