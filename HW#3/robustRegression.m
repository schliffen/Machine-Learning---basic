function [model] = robustRegression(X,y)

[n,d] = size(X);

% Initial guess
w = zeros(d,1);

% This is how you compute the function and gradient:
[f,g] = funObj(w,X,y);

% Derivative check that the gradient code is correct:
[f2,g2] = autoGrad(w,@funObj,X,y);

if max(abs(g-g2) > 1e-4)
    fprintf('User and numerical derivatives differ:\n');
    [g g2]
else
    fprintf('User and numerical derivatives agree.\n');
end

% Solve least squares problem
w = findMin(@funObj,w,100,X,y);

model.w = w;
model.predict = @predict;

end

function [yhat] = predict(model,Xtest)
w = model.w;
yhat = Xtest*w;
end

function [f,g] = funObj(w,X,y)

f = 0;
for i = 1:size(X, 1)
    f = f + log(exp(w.*X(i, :)-y(i))+exp(-w.*X(i, :)+y(i)));
end
g = 0;
for i = 1:size(X, 1)
    g = g + (X(i, :)'.*exp(w.*X(i, :)-y(i))-X(i, :)'.*exp(-w.*X(i, :)+y(i)))/...
        (exp(w.*X(i, :)-y(i))+exp(-w.*X(i, :)+y(i)));
end

end