function [model] = dimRedRPCA(X,k)

[n,d] = size(X);

% Subtract mean
mu = mean(X);
X = X - repmat(mu,[n 1]);

% Initialize W and Z
W = randn(k,d);
Z = randn(n,k);

R = Z*W-X;
f = sum(sum(R.^2));
for iter = 1:50
    fOld = f;
    
    % Update Z
    Z(:) = findMin(@funObjZ,Z(:),10,0,X,W);
    
    % Update W
    W(:) = findMin(@funObjW,W(:),10,0,X,Z);
    
    R = Z*W-X;
    f = sum(sum(R.^2));
    fprintf('Iteration %d, loss = %.5e\n',iter,f);
    
    if fOld - f < 1e-4
        break;
    end
end

model.mu = mu;
model.W = W;
model.compress = @compress;
model.expand = @expand;
end

function [Z] = compress(model,X)
[t,d] = size(X);
mu = model.mu;
W = model.W;
k = size(W,1);

X = X - repmat(mu,[t 1]);
% We didn't enforce that W was orthogonal so we need to optimize to find Z
Z = zeros(t,k);
Z(:) = findMin(@funObjZ,Z(:),100,0,X,W);
end

function [X] = expand(model,Z)
[t,d] = size(Z);
mu = model.mu;
W = model.W;

X = Z*W + repmat(mu,[t 1]);
end

function [f,g] = funObjW(W,X,Z)
% Resize vector of parameters into matrix
d = size(X,2);
k = size(Z,2);
W = reshape(W,[k d]);



epsil = 1e-4;

R = Z*W-X;
f = sum(sum(abs(R)));

g = zeros(size(W));
for p = 1:k
    for q = 1:d
        for i = 1:size(X, 1)
            g(p, q) = g(p, q) + (Z(i, p)*W(p, q) - X(i, q))*Z(i, p)/ ...
                sqrt((Z(i, p)*W(p, q) - X(i, q))^2 + epsil);
        end
    end
end
      
% Return a vector
g = g(:);
end

function [f,g] = funObjZ(Z,X,W)
% Resize vector of parameters into matrix
n = size(X,1);
k = size(W,1);
Z = reshape(Z,[n k]);

epsil = 1e-4;
% Compute function value
R = Z*W-X;
f = sum(sum(abs(R)));
g = zeros(size(Z));
for p = 1:n
    for q = 1:k
        for j = 1:size(X, 2)
            g(p, q) = g(p, q) + (Z(p, q)*W(q, j) - X(p, j))*W(q, j)/ ...
                sqrt((Z(p, q)*W(q, j) - X(p, j))^2 + epsil);
        end
    end
end
         
% Return a vector
g = g(:);
end
