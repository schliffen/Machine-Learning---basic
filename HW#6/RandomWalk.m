function PR = RandomWalk(H, itr)


n = size(H, 1);
k = 0;
v = ceil(n*rand);
PR = zeros(n, 1);

while (k <= itr)
    
    alpha = 0.1;
    
    if sum(H(v, :)) == 0
        alpha = 1;
    end
    
    if (rand <= alpha)
        v = ceil(n*rand);        
    else 
        ind = find(H(v, :));
        v = ind(ceil(length(ind)*rand));
    end
    
    k = k + 1;
        
    if k>100
        PR(v) = PR(v) + 1;
    end

end

PR = PR/sum(PR);

end