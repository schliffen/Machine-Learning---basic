function [model] = weightedLeastSquares(X,y,z)
Z = diag(z);

% Solve least squares problem
w = (X'*Z*X)\X'*Z*y;

model.w = w;
model.predict = @predict;

end

function [yhat] = predict(model,Xhat)
w = model.w;
yhat = Xhat*w;
end