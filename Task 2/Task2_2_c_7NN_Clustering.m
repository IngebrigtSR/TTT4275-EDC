%%Clustering Data
path = addpath('D:\OneDrive\Dokumenter\NTNU\Estimering Deteksjon og Klassifisering\Classification project\Task 2');
if exist('trainvClust','var')
    
else
    run('Task2_2_a_Clustering.m');
end
disp('Data Clustered');

%% Defining some values
Ntest       = size(testv,1);
Ntrain      = size(trainvClust,1);
chunkSize   = 1000;
Nchunks     = Ntrain / chunkSize;
Nclasses    = 10;
k           = 7;


%% Classifier
guess = zeros(Nclasses,Ntest);
disp('7-NN Classification with clustering');
tic;
for i = 1:Ntest
    distances = dist(trainvClust,testv(i,:)');
    [~,idx] = sort(distances);
    k_min = idx(1:k);
    labels = trainlabClust(k_min);
    num = 0:9;
    count = hist(labels,num);
    [~,pred] = max(count);
    guess(pred,i) = 1;
end
toc

%% Knowns for Confusion Matrix
known = zeros(Nclasses,Ntest);
for i = 1:Ntest
    l = testlab(i);
    known(l+1,i) = 1;
end


%% Confusion
plotconfusion(known,guess);
tag = get(get(gca,'title'),'string');
title({tag, '7-NN, Using Clustering'});
xticklabels({'0','1','2','3','4','5','6','7','8','9'});
yticklabels({'0','1','2','3','4','5','6','7','8','9'});