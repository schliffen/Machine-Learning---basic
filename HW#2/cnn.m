function [model] = cnn(X,y,k)

% Implementation of condensed k-nearest neighbour classifer
S(1, 1:2) = X(1, 1:2);
y_s = y(1);
j = 2;

for i = 2:size(X, 1)
    
    [model] = knn(S, y_s, k);
    yhat_test = model.predict(model,X(i, :));
    error_test = (y(i) ~= yhat_test);
    
    if error_test == 1
        S(j, 1:2) = X(i, :);
        y_s(j, 1) = y(i);
        j = j + 1;
    end
end


model.X = S;
model.y = y_s;
model.k = k;
model.c = max(y);
model.predict = @predict;

end

function [yhat] = predict(model,Xtest)
% Write me!

X = model.X;
y = model.y;
k = model.k;

[n,d] = size(X);
[t,d] = size(Xtest);

D = X.^2*ones(d, t) + ones(n, d)*(Xtest').^2 - 2*X*Xtest';
D = D.^.5;
yhat = zeros(t, 1);

for i = 1:t
    
    [tmp, ind] = sort(D(:, i));
    yhat(i) = mode(y(ind(1:k)));
end
    

end