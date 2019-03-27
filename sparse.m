clear
load('ad_data.mat');

% Add a column of 1 to add the bias 
X_train = [ones(size(X_train, 1), 1) X_train];
X_test = [ones(size(X_test, 1), 1) X_test];

%list of regularization parameters
parameters = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];

%initialize number of feature and Area under the curve
n_feature = zeros(size(parameters));
area = zeros(size(parameters));
count = 1;


for par = parameters
    [weights, bias] = logistic_l1_train(X_train, y_train, par);
    n_feature(count) = nnz(weights);
    predictions = X_test * weights;
    [X, Y, T, AUC] = perfcurve(y_test, predictions, 1);
    area(count) = AUC;
    count = count + 1;
end

plot(parameters, n_feature)
title(' number of feature selected vs. Regularization Parameter')
xlabel(' Parameter')
ylabel('number of feature selected ')

figure
plot(parameters, area)
title(' AUC vs. Regularization Parameter')
xlabel(' Parameter')
ylabel('AUC')
