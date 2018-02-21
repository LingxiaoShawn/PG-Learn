function data = LoadData(dataset, noise_level, train_per, partition, class_set)
% Load data from dataset.mat file, and split it into train, validation and
% test sets. 
% ============
% Input: 
%   dataset - name of dataset
%   noise_level - the percentage of added noise columns
%   train_per - training set percentage, the value can be
%               0.1 or 0.2 or 0.3 or 0.4 or 0.5
%   partition - partition number (1 - 10) corresponding to a partition index file
%   class_set - a list of classes to classify
%     
% Output: 
%   data: processed data, divided into labeled data and unlabeled data.
%   | data.XTrain - n*d feature matrix for training data
%   | data.yTrain - n*1 class labels for training data (label starts from 0)
%   | data.XValid - n*d feature matrix for validation data
%   | data.yValid - n*1 class labels for validation data (label starts from 0)
%   | data.XTest - n*d feature matrix for test data
%   | data.yTest - n*1 class labels for test data (label starts from 0) 
%   -------------------------------------------------------------------
file_name = mfilename();
file_path = mfilename('fullpath');
root_path = file_path(1: end - size(file_name,2) - 1 - size('util',2));

if ispc()
    slash = '\';
else
    slash = '/';
end

% load data mat file
load(strcat(root_path,'datasets',slash,dataset,slash,dataset,'.mat'));

% add noise
fea = [fea, rand(size(fea,1), ceil(size(fea,2)/100*noise_level))];

% load partition index file
load(strcat(root_path,'datasets',slash,dataset,slash,num2str(train_per * 100),...
    'PerTrain',slash,num2str(partition),'.mat'));

if nargin < 5
    if strcmp(dataset, 'coil')
        class_set = 0:5;
    elseif strcmp(dataset, 'usps')
        class_set = 0:9;
    elseif strcmp(dataset, 'mnist')
        class_set = 0:9;
    elseif strcmp(dataset, 'orl')
        class_set = 1:40;
    elseif strcmp(dataset, 'yale')
        class_set = 1:5;
    elseif strcmp(dataset, 'umist')
        class_set = 1:20;
    end
end

XTrain = fea(trainIdx,:);  %#ok<NODEF>
yTrain = gnd(trainIdx,:);  %#ok<NODEF>

XTest = fea(testIdx,:);  %#ok<NODEF>
yTest = gnd(testIdx,:); 

% select some classes
trainIdx = ismember(yTrain, class_set);
yTrain = yTrain(trainIdx,:);
XTrain = XTrain(trainIdx,:);

testIdx = ismember(yTest, class_set);
yTest = yTest(testIdx,:);
XTest = XTest(testIdx,:);

% transform labels to start from zero
for i = 1:size(class_set,2)
    yTrain(yTrain == class_set(i)) = i - 1;
    yTest(yTest == class_set(i)) = i - 1;
end

idx = find(yTrain == 1, 1);
end_idx = find(yTrain == 0);
l = end_idx(idx) - 1;

XValid = XTrain(l+1:end,:);
yValid = yTrain(l+1:end); 
XTrain = XTrain(1:l,:);
yTrain = yTrain(1:l); 

data.XTrain = XTrain;
data.yTrain = yTrain;
data.XValid = XValid;
data.yValid = yValid;
data.XTest = XTest;
data.yTest = yTest;

end