function [model] = logRegL2(X,y, lambda)

[n,d] = size(X);

maxFunEvals = 400; % Maximum number of evaluations of objective
verbose = 1; % Whether or not to display progress of algorithm
w0 = zeros(d,1);
model.w = findMin(@logisticLoss,w0,maxFunEvals,verbose,X,y, lambda);
model.predict = @(model,X)sign(X*model.w); % Predictions by taking sign
end

function [f,g] = logisticLoss(w,X,y, lambda)
yXw = y.*(X*w);
f = sum(log(1 + exp(-yXw))) + lambda/2*(w'*w); % Function value
g = -X'*(y./(1+exp(yXw))) + lambda*w; % Gradient
end