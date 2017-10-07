% Z = randn(n, k);
% 
% for i = 1:50
%     
%     W = (Z'*Z)^-1*Z'*X;
%     Z = X*W'*(W*W')^-1;
% end
