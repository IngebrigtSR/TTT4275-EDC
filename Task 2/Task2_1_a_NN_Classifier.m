%Task 2_1

%%Making sure variables are loaded
%Replace the path with whatever folder you've stored read09.m in
path = addpath('D:\OneDrive\Dokumenter\NTNU\Estimering Deteksjon og Klassifisering\Classification project\Task 2\MNist_ttt4275');
if exist('testv','var')
    
else
    run('read09.m');
end
disp('Data loaded');

%% Defining some values
Ntest     = size(testv,1);
Ntrain    = size(trainv,1);
Nclasses    = 10;

%% Classifier
guess = zeros(Nclasses,Ntest);
disp('1-NN Classification');
tic;
for i = 1:Ntest
    distances = dist(trainv,testv(i,:)');
    [d,dis] = min(distances);
    predicted = trainlab(dis);
    guess(predicted+1,i) = 1;
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
title({tag, '1-NN, no clustering'});
xticklabels({'0','1','2','3','4','5','6','7','8','9'});
yticklabels({'0','1','2','3','4','5','6','7','8','9'});