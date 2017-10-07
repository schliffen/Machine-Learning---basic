function [Iquant] = quantizeImage(I,b)

k = 2^b;
% X = zeros(size(I, 1)*size(I, 2), 3);

X = reshape(I, size(I, 1)*size(I, 2), 3);

doPlot = 0;
model = clusterKmeans(X,k,doPlot);
y = model.predict(model,X);

Inew = zeros(length(y), 3);
for i = 1:length(y);
    Inew(i, :) = model.W(y(i), :);
end


Iquant = round(reshape(Inew, size(I, 1), size(I, 2), size(I, 3)));


end
