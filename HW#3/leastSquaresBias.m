function [model] = leastSquaresBias(X,y)

X = [ones(size(X, 1), 1) X];
% Solve least squares problem
w = (X'*X)\X'*y;

model.w = w;
model.predict = @predict;

end

function [yhat] = predict(model,Xhat)
Xhat = [ones(size(Xhat, 1), 1) Xhat];
w = model.w;
yhat = Xhat*w;
end