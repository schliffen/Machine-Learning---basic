function [Z] = visualizeISOMAP(X,k,names)

[n,d] = size(X);

% Compute all distances
D = X.^2*ones(d,n) + ones(n,d)*(X').^2 - 2*X*X';
D = sqrt(abs(D));

G = zeros(size(D));
for i = 1:size(D, 1)
    [~, I] = sort(D(i, :));
    G(i, I(1:(k+1))) = D(i, I(1:(k+1)));
end

for i = 1:size(D, 1)
    for j = 1:size(D, 2)
        if G(i, j) ~= 0
            G(j, i) = G(i, j);
        end
    end
end


DD = zeros(size(D));
for i = 1:size(D, 1)
    for j = 1:size(D, 2)
        [L, ~] = dijkstra(G,i,j);
        DD(i, j) = L;
    end
end

vecD = DD(:);
vecD = sort(vecD, 'descend');
ind = find(vecD ~= inf, 1);
MaxDD = vecD(ind);
for i = 1:size(DD, 1)
    for j = 1:size(DD, 2)
        if DD(i, j) == inf
            DD(i, j) = MaxDD;
        end
    end
end

% Initialize low-dimensional representation with PCA
model = dimRedPCA(X,2);
Z = model.compress(model,X);

Z(:) = findMin(@stress,Z(:),500,0,DD,names);

end

function [f,g] = stress(Z,D,names)

n = length(D);
k = numel(Z)/n;

Z = reshape(Z,[n k]);

f = 0;
g = zeros(n,k);
for i = 1:n
    for j = i+1:n
        % Objective Function
        Dz = norm(Z(i,:)-Z(j,:));
        s = D(i,j) - Dz;
        f = f + (1/2)*s^2;
        
        % Gradient
        df = s;
        dgi = (Z(i,:)-Z(j,:))/Dz;
        dgj = (Z(j,:)-Z(i,:))/Dz;
        g(i,:) = g(i,:) - df*dgi;
        g(j,:) = g(j,:) - df*dgj;
    end
end
g = g(:);

% Make plot if using 2D representation
if k == 2
    figure(3);
    clf;
    plot(Z(:,1),Z(:,2),'.');
    if ~isempty(names)
        hold on;
        for i = 1:n
            text(Z(i,1),Z(i,2),names(i,:));
        end
    end
    pause(.01)
end
end