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