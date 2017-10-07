clear; clc; close all

load arrowhead.mat

n = length(names);
alpha = .1;

%% The matrix

H = sparse(X(:, 1), X(:, 2), 1, n, n);
H = full(H);


for i = 1:n
    for j = 1:n
        if H(i, j) ~= 0
            H(j, i) = H(i, j);
        end
    end
end


%% Random Walk
itr = 1e6;
PR = RandomWalk(H, itr);

%%
bar(PR)
xlim([0, 105])
title('Exercise #1 a - Bar graph of the PageRank')
xlabel('Participants', 'FontSize', 12, 'FontWeight','bold')
ylabel('PageRank', 'FontSize', 12, 'FontWeight','bold')
grid on
