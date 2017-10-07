
% Load data
load outliersData.mat

% Plot data
figure(1);
plot(X,y,'b.')
title('Training Data');
hold on

z = ones(size(X, 1), 1);
z(401:end) = 0.1;

% Fit least-squares estimator
model = weightedLeastSquares(X,y,z);

% Draw model prediction
Xsample = [min(X):.01:max(X)]';
yHat = model.predict(model,Xsample);
plot(Xsample,yHat,'g-');
