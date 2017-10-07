function [model] = leastSquaresBasis(x, y, p)

X = polyBasis(x, p);

% Solve least squares problem
w = (X'*X)\X'*y;

model.w = w;
model.predict = @predict;

end

function [yhat] = predict(model,xhat)
w = model.w;
p = length(w)-2;
Xhat = polyBasis(xhat, p);

yhat = Xhat*w;
end