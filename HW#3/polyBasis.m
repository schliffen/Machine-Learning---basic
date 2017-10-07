function X = polyBasis(x, p)

X = ones(length(x), p+2);

for i = 2:p+1
    X(:, i) = x.^(i-1);
end
X(:, end) = sin(2*pi*x/1.258);
end