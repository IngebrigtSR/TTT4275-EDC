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


%% Classifier
guess = zeros(Nclasses,Ntest);
disp('1-NN Classification with clustering');
tic;
for i = 1:Ntest
    distances = dist(trainvClust,testv(i,:)');
    [d,dis] = min(distances);
    guess(trainlabClust(dis)+1,i) = 1;
end
toc

%% Knowns for Confusion Matrix
known = zeros(Nclasses,Ntest);
for i = 1:Ntest
    known(testlab(i)+1,i) = 1;
end


%% Confusion
plotconfusion(known,guess);
tag = get(get(gca,'title'),'string');
title({tag, '1-NN, Using Clustering'});
xticklabels({'0','1','2','3','4','5','6','7','8','9'});
yticklabels({'0','1','2','3','4','5','6','7','8','9'});