function [y] = predict_ifelse(model,X)

[t,d] = size(X);
y = zeros(t,1);

for i = 1:t
    if X(i,model.splitModel.splitVariable) > model.splitModel.splitValue
        if X(i,model.subModel1.splitVariable) > model.subModel1.splitValue
            y(i,1) = model.subModel1.splitSat;
        else
            y(i,1) = model.subModel1.splitNot;
        end
    else
        if X(i,model.subModel0.splitVariable) > model.subModel0.splitValue
            y(i,1) = model.subModel0.splitSat;
        else
            y(i,1) = model.subModel0.splitNot;
        end
    end
end

end
