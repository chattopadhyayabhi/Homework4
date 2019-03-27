function [weights] = mylogistic_train(data, labels)

    epsilon = 1e-5;
    maxiter = 1000;


weights = zeros(size(data, 2), 1);
%iterative newton Raphson
for i = 1:maxiter
    y1 = sigmf(data * weights, [1 0]);
    R = diag(y1 .* (1 - y1));
    z = data * weights - ((R +0.1*eye(size(R, 1)))^-1) * (y1 - labels);
    weights = ((data' * R * data)^-1) * data' * R * z;
    y2 = sigmf(data * weights, [1 0]);
    
    % iteration end criteria
    if mean(abs(y1 - y2)) < epsilon
        break
    end
end