load animals.mat

[n,d] = size(X);
X = standardizeCols(X);

figure(1);
imagesc(X);
figure(2);
i = ceil(rand*d);
j = ceil(rand*d);

[model] = dimRedPCA(X,14);
W = model.W;
Z = X*W'*(W*W')^-1;

plot(Z(:,1),Z(:,2),'.');
% gname(animals);
    
1- norm((Z*W - X), 'fro')/norm(X, 'fro')