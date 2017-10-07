function [model] = decisionStump(X,y)

[n,d] = size(X);

y_mode = mode(y);
minError = sum(y ~= y_mode);
splitVariable = [];
splitValue = [];
splitSat = y_mode;
splitNot = [];

if any(y ~= y(1))
   
    for j = 1:d
        for i = 1:n
            
            value = X(i,j);
            
            y_sat = mode(y(X(:,j) > value));
            y_not = mode(y(X(:,j) <= value));
            
            yhat = y_sat*ones(n,1);
            yhat(X(:,j) <= value) = y_not;
            
            error = sum(yhat ~= y);

            if error < minError
               
                minError = error;
               splitVariable = j;
               splitValue = value;
               splitSat = y_sat;
               splitNot = y_not;
               
            end
        end
    end
end

model.splitVariable = splitVariable;
model.splitValue = splitValue;
model.splitSat = splitSat;
model.splitNot = splitNot;
model.predict = @predict;

end


function [y] = predict(model,X)

[t,d] = size(X);

if isempty(model.splitVariable)
    
    y = model.splitSat*ones(t,1);

else
    
    y = zeros(t,1);
    
    for i = 1:t
        
        if X(i,model.splitVariable) > model.splitValue
            
            y(i,1) = model.splitSat;
            
        else
            
            y(i,1) = model.splitNot;
            
        end
    end
end

end
