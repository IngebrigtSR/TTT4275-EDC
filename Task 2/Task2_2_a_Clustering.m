M = 64;
clusters = cell(10,1);
trainvectors = cell(10,1);

for i = 0:9
    trainvectors{i+1} = trainv(trainlab == i,:);
end

for i = 1:10
    [~,Ci] = kmeans(trainvectors{i},M);
    clusters{i} = Ci;
end

trainvClust = cell2mat(clusters);
trainlabClust = NaN(10*M,1);
for i = 0:9
    trainlabClust(i*M+1:(i+1)*M) = i * ones(M,1);
end