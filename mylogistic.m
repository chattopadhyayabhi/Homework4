
data = load('data.txt');
%bias term inclusion
data = [ones(size(data, 1), 1), data];
labels = load('labels.txt');

%  training data uses the first 2000 rows
train_labels = labels(1:2000);
%testing data uses the rest
test_data = data(2001:end, :);
test_labels = labels(2001:end);


n = [200, 500, 800, 1000, 1500, 2000];

count = [];

for i = n
    weights = mylogistic_train(data(1:i, :), labels(1:i));
    prediction =sigmf( test_data * weights,[1,0]);
    
    correct_pred = (test_labels == prediction);
    accuracy = sum(correct_pred) / numel(correct_pred);
    count = [count, accuracy];
end

plot(amounts, count, '-d')
title(' Accuracy with respect to n')
xlabel('n')
ylabel(' accuracy')
