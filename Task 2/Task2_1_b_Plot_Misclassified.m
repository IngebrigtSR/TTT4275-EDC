%%Adding path to function folder
addpath('D:\OneDrive\Dokumenter\NTNU\Estimering Deteksjon og Klassifisering\Classification project\Task 2\functions');
load('data_all.mat');

%% Locating Misclassified
Nmisclass = 10;
misclassifiedID = NaN(Nmisclass,1);
for i = 1:length(guess)
    g = find(guess(:,i));
    k = find(known(:,i));
    
    if g ~= k && isnan(misclassifiedID(g))
        misclassifiedID(g) = i;
    end
end

%% Drawing Misclassified
figure(1);
clf;
sgtitle('Misclassified images');

for i = 1:Nmisclass
    subplot(2,5,i); %num1 x num2 = Nmisclass (as a general rule, but this code hasn't really been scaled for anything other than 10)
    set(gca,'XTick',[],'YTick',[])
    id = misclassifiedID(i);
    imagedrawer(testv(id,:));
    title({sprintf('Guessed: %d',i-1),
            sprintf('Actual: %d',testlab(id))});
end
