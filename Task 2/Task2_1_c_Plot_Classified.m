%%Adding path to function folder
addpath('D:\OneDrive\Dokumenter\NTNU\Estimering Deteksjon og Klassifisering\Classification project\Task 2\functions');
load('data_all.mat');

%% Locating Correctly Classified
Nclass = 10;
classifiedID = NaN(Nclass,1);
for i = 1:length(guess)
    g = find(guess(:,i));
    k = find(known(:,i));
    
    if g == k && isnan(classifiedID(g))
        classifiedID(g) = i;
    end
end

%% Drawing Classified
figure(1);
clf;
sgtitle('Correctly Classified Images');

for i = 1:Nclass
    subplot(2,5,i); %num1 x num2 = Nmisclass (as a general rule, but this code hasn't really been scaled for anything other than 10)
    set(gca,'XTick',[],'YTick',[])
    id = classifiedID(i);
    imagedrawer(testv(id,:));
    title({sprintf('Guessed: %d',i-1),
            sprintf('Actual: %d',testlab(id))});
end