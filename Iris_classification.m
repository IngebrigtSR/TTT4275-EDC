%Iris task

%%Part 1
%All the parameters we are going to need
C=3; %number of classes
D=4; %number of numeric attributes
W=zeros(C,D);%the matrix of the classes with the average attributes
w=zeros(C,1);%the vector of the classes
Wall=[W w];
samples=1000; %amount of samples we are going to use
MSE_all=zeros(1,samples); %Minimum Square error for each sample
MSE_grad_all=zeros(1,samples); %The gradient of each MSE, we need it to calculate the equation (23)
t_1=[1; 0; 0]; %the target for the first class
t_2=[0; 1; 0]; %target for class 2
t_3=[0; 0; 1]; %target for class 3

%all the parameters that we can change
Ntrain=30; %number of samples for the training sequence
Ntest=20; %number of samples for the test sequence
Ntot=Ntrain+Ntest;
alpha=0.0075; %step factor that should be found while running simulation

%We load all the classes with the attributes
x1all = load('class_1','-ascii');
x2all = load('class_2','-ascii');
x3all = load('class_3','-ascii');
xall=[x1all; x2all; x3all];


%The training sets
x1_train=x1all(1:Ntrain,:); %train class 1
x2_train=x2all(1:Ntrain,:); %class 2
x3_train=x3all(1:Ntrain,:); %class 3
x_train=[x1_train; x2_train; x3_train];%easier to put it like this, we will be able to process everything in a loop

%For the other set
% x1_train=x1all(Ntest+1:end,:); %train class 1
% x2_train=x2all(Ntest+1:end,:); %class 2
% x3_train=x3all(Ntest+1:end,:); %class 3
% x_train=[x1_train; x2_train; x3_train];%easier to put it like this, we will be able to process everything in a loop

%the test sets
x1_test=x1all(Ntrain+1:end,:);
x2_test=x2all(Ntrain+1:end,:);
x3_test=x3all(Ntrain+1:end,:);
x_test=[x1_test; x2_test; x3_test];

% x1_test=x1all(1:Ntest,:);
% x2_test=x2all(1:Ntest,:);
% x3_test=x3all(1:Ntest,:);
% x_test=[x1_test; x2_test; x3_test];

%We will have to update W until W(m)-W(m-1) is small enough (reference to
%the equation (23) in the compendium)

%training
for i=1:samples
    %for the first class
    MSE=0;
    MSE_grad=0;
    for k=1:size(x_train,1) %until 90
        if k<=Ntrain %corresponds to the first class
            tk=t_1;
        elseif k<=2*Ntrain %second class
            tk=t_2;
        else %and third class
            tk=t_3;
        end
        xk=[x_train(k,:)';1]; %to have a good dimension to multiply by W
        %sigmoid calculation
        zk=Wall*xk+w;
        gk=1./(1+exp(-zk));
        MSE_grad=MSE_grad+((gk-tk).*(gk).*(1-gk))*xk';
        MSE=MSE+0.5*((gk-tk)'*(gk-tk));
    end
    MSE_grad_all(i)=norm(MSE_grad);
    MSE_all(i)=MSE;
    Wall=Wall-alpha*MSE_grad;
    
end 

%confusion matrix for training
conf_train=zeros(C,C); %matrix 3x3
 for k=1:size(x_train,1) %until 90
        if k<=Ntrain %corresponds to the first class
            tk=t_1;
            class_trained=1;
        elseif k<=2*Ntrain %second class
            tk=t_2;
            class_trained=2;
        else %and third class
            tk=t_3;
            class_trained=3;
        end
        xk=[x_train(k,:)';1]; %to have a good dimension to multiply by W
        %sigmoid calculation
        zk=Wall*xk+w;
        gk=1./(1+exp(-zk));
        [gmax, imax]=max(gk);
        conf_train(class_trained,imax)=conf_train(class_trained,imax)+1;
 end
disp('confusion matrix for training=');
disp(conf_train);

%confusion matrix for testing
conf_test=zeros(C,C); %matrix 3x3
 for k=1:size(x_test,1) %until 90
        if k<=Ntest %corresponds to the first class
            tk=t_1;
            class_tested=1;
        elseif k<=2*Ntest %second class
            tk=t_2;
            class_tested=2;
        else %and third class
            tk=t_3;
            class_tested=3;
        end
        xk=[x_test(k,:)';1]; %to have a good dimension to multiply by W
        %sigmoid calculation
        zk=Wall*xk+w;
        gk=1./(1+exp(-zk));
        [gmax, imax]=max(gk);
        conf_test(class_tested,imax)=conf_test(class_tested,imax)+1;
 end
disp('confusion matrix for testing=');
disp(conf_test);

%error rate for training
error_r_train=0;
for i=1:C
    for j=1:C
        if i~=j
            error_r_train=error_r_train+conf_train(i,j);
        end
    end
end
error_r_train=error_r_train/(Ntrain*C); %error rate is in percentage
disp('error rate for training=');
disp(error_r_train);

%error rate for testing
error_r_test=0;
for ib=1:C
    for jb=1:C
        if ib~=jb
            error_r_test=error_r_test+conf_test(ib,jb);
        end
    end
end
error_r_test=error_r_test/(Ntest*C); %error rate is in percentage
disp('error rate for testing=');
disp(error_r_test);

%%Part 2
%Histogram for each features
% nbin_feature1=34; %max =7.7 and min=4.3
% nbin_feature2=24; %max=4.4 min=2
% nbin_feature3=59%max=6.9 min=1
% nbin_feature=24 %max=2.5 max=0.1
% %the interval to study is [0;8]

%Class1
x_feat1 = x1all(:,1);
x_feat2 = x1all(:,2);
x_feat3 = x1all(:,3);
x_feat4 = x1all(:,4);
figure(1);
subplot(3,4,1);
histogram(x_feat1);
set(gca,'XLim',[0 8]);  % set x-axis limits between 0-10  
title('Class 1 Sepal length ');

figure(1);
subplot(3,4,2);
histogram(x_feat2);
set(gca,'XLim',[0 8]);  % set x-axis limits between 0-10  
title('Class 1 Sepal width');

figure(1);
subplot(3,4,3);
histogram(x_feat3);
set(gca,'XLim',[0 8]);  % set x-axis limits between 0-10  
title('Class 1 Petal length');

figure(1);
subplot(3,4,4);
histogram(x_feat4);
set(gca,'XLim',[0 8]);  % set x-axis limits between 0-10  
title('Class 1 Petal width');

%Class2
x_feat1 = x2all(:,1);
x_feat2 = x2all(:,2);
x_feat3 = x2all(:,3);
x_feat4 = x2all(:,4);
figure(1);
subplot(3,4,5);
histogram(x_feat1);
set(gca,'XLim',[0 8]);  % set x-axis limits between 0-10  
title('Class 2 Sepal length ');

figure(1);
subplot(3,4,6);
histogram(x_feat2);
set(gca,'XLim',[0 8]);  % set x-axis limits between 0-10  
title('Class 2 Sepal width');

figure(1);
subplot(3,4,7);
histogram(x_feat3);
set(gca,'XLim',[0 8]);  % set x-axis limits between 0-10  
title('Class 2 Petal length');

figure(1);
subplot(3,4,8);
histogram(x_feat4);
set(gca,'XLim',[0 8]);  % set x-axis limits between 0-10  
title('Class 2 Petal width');

%Class3
x_feat1 = x3all(:,1);
x_feat2 = x3all(:,2);
x_feat3 = x3all(:,3);
x_feat4 = x3all(:,4);
figure(1);
subplot(3,4,9);
histogram(x_feat1);
set(gca,'XLim',[0 8]);  % set x-axis limits between 0-10  
title('Class 3 Sepal length ');

figure(1);
subplot(3,4,10);
histogram(x_feat2);
set(gca,'XLim',[0 8]);  % set x-axis limits between 0-10  
title('Class 3 Sepal width');

figure(1);
subplot(3,4,11);
histogram(x_feat3);
set(gca,'XLim',[0 8]);  % set x-axis limits between 0-10  
title('Class 3 Petal length');

figure(1);
subplot(3,4,12);
histogram(x_feat4);
set(gca,'XLim',[0 8]);  % set x-axis limits between 0-10  
title('Class 3 Petal width');
%We should remove sepal width so the 2nd feature

%training and testing without this feature
%training
D=3;
W_1feat=zeros(C,D);%the matrix of the classes with the average attributes
w_1feat=zeros(C,1);%the vector of the classes
Wall_1feat=[W_1feat w_1feat];
MSE_all_1feat=zeros(1,samples);
MSE_grad_all_1feat=zeros(1,samples); %The gradient of each MSE, we need it to calculate the equation (23)

x_train_1feat=[x_train(:,1) x_train(:,3:4)];
x_test_1feat=[x_test(:,1) x_test(:,3:4)];
for i=1:samples
    %for the first class
    MSE=0;
    MSE_grad=0;
    for k=1:size(x_train_1feat,1) %until 90
        if k<=Ntrain %corresponds to the first class
            tk=t_1;
        elseif k<=2*Ntrain %second class
            tk=t_2;
        else %and third class
            tk=t_3;
        end
        xk=[x_train_1feat(k,:)';1]; %to have a good dimension to multiply by W
        %sigmoid calculation
        zk=Wall_1feat*xk+w_1feat;
        gk=1./(1+exp(-zk));
        MSE_grad=MSE_grad+((gk-tk).*(gk).*(1-gk))*xk';
        MSE=MSE+0.5*((gk-tk)'*(gk-tk));
    end
    MSE_grad_all_1feat(i)=norm(MSE_grad);
    MSE_all_1feat(i)=MSE;
    Wall_1feat=Wall_1feat-alpha*MSE_grad;
    
end 

%confusion matrix for training
conf_train_1feat=zeros(C,C); %matrix 3x3
 for k=1:size(x_train_1feat,1) %until 90
        if k<=Ntrain %corresponds to the first class
            tk=t_1;
            class_trained=1;
        elseif k<=2*Ntrain %second class
            tk=t_2;
            class_trained=2;
        else %and third class
            tk=t_3;
            class_trained=3;
        end
        xk=[x_train_1feat(k,:)';1]; %to have a good dimension to multiply by W
        %sigmoid calculation
        zk=Wall_1feat*xk+w_1feat;
        gk=1./(1+exp(-zk));
        [gmax, imax]=max(gk);
        conf_train_1feat(class_trained,imax)=conf_train_1feat(class_trained,imax)+1;
 end
disp('confusion matrix for training with just 3 features=');
disp(conf_train_1feat);

%confusion matrix for testing
conf_test_1feat=zeros(C,C); %matrix 3x3
 for k=1:size(x_test_1feat,1) %until 90
        if k<=Ntest %corresponds to the first class
            tk=t_1;
            class_tested=1;
        elseif k<=2*Ntest %second class
            tk=t_2;
            class_tested=2;
        else %and third class
            tk=t_3;
            class_tested=3;
        end
        xk=[x_test_1feat(k,:)';1]; %to have a good dimension to multiply by W
        %sigmoid calculation
        zk=Wall_1feat*xk+w_1feat;
        gk=1./(1+exp(-zk));
        [gmax, imax]=max(gk);
        conf_test_1feat(class_tested,imax)=conf_test_1feat(class_tested,imax)+1;
 end
disp('confusion matrix for testing with just 3 features=');
disp(conf_test_1feat);

%error rate for training
error_r_train_1feat=0;
for i=1:C
    for j=1:C
        if i~=j
            error_r_train_1feat=error_r_train_1feat+conf_train_1feat(i,j);
        end
    end
end
error_r_train_1feat=error_r_train_1feat/(Ntrain*C); %error rate is in percentage
disp('error rate for training with 3 features=');
disp(error_r_train_1feat);

%error rate for testing
error_r_test_1feat=0;
for ib=1:C
    for jb=1:C
        if ib~=jb
            error_r_test_1feat=error_r_test_1feat+conf_test_1feat(ib,jb);
        end
    end
end
error_r_test_1feat=error_r_test_1feat/(Ntest*C); %error rate is in percentage
disp('error rate for testing with just 3 features=');
disp(error_r_test_1feat);

%If we want to keep just 2 features, we are going to remove sepal width and
%sepal length (1st features)

%2 Features
%training and testing without this feature
%training
D=2;
W_2feat=zeros(C,D);%the matrix of the classes with the average attributes
w_2feat=zeros(C,1);%the vector of the classes
Wall_2feat=[W_2feat w_2feat];
MSE_all_2feat=zeros(1,samples);
MSE_grad_all_2feat=zeros(1,samples); %The gradient of each MSE, we need it to calculate the equation (23)

x_train_2feat=x_train(:,3:4);
x_test_2feat=x_test(:,3:4);
for i=1:samples
    %for the first class
    MSE=0;
    MSE_grad=0;
    for k=1:size(x_train_2feat,1) %until 90
        if k<=Ntrain %corresponds to the first class
            tk=t_1;
        elseif k<=2*Ntrain %second class
            tk=t_2;
        else %and third class
            tk=t_3;
        end
        xk=[x_train_2feat(k,:)';1]; %to have a good dimension to multiply by W
        %sigmoid calculation
        zk=Wall_2feat*xk+w_2feat;
        gk=1./(1+exp(-zk));
        MSE_grad=MSE_grad+((gk-tk).*(gk).*(1-gk))*xk';
        MSE=MSE+0.5*((gk-tk)'*(gk-tk));
    end
    MSE_grad_all_2feat(i)=norm(MSE_grad);
    MSE_all_2feat(i)=MSE;
    Wall_2feat=Wall_2feat-alpha*MSE_grad;
    
end 

%confusion matrix for training
conf_train_2feat=zeros(C,C); %matrix 3x3
 for k=1:size(x_train_2feat,1) %until 90
        if k<=Ntrain %corresponds to the first class
            tk=t_1;
            class_trained=1;
        elseif k<=2*Ntrain %second class
            tk=t_2;
            class_trained=2;
        else %and third class
            tk=t_3;
            class_trained=3;
        end
        xk=[x_train_2feat(k,:)';1]; %to have a good dimension to multiply by W
        %sigmoid calculation
        zk=Wall_2feat*xk+w_2feat;
        gk=1./(1+exp(-zk));
        [gmax, imax]=max(gk);
        conf_train_2feat(class_trained,imax)=conf_train_2feat(class_trained,imax)+1;
 end
disp('confusion matrix for training with just 2 features=');
disp(conf_train_2feat);

%confusion matrix for testing
conf_test_2feat=zeros(C,C); %matrix 3x3
 for k=1:size(x_test_2feat,1) %until 90
        if k<=Ntest %corresponds to the first class
            tk=t_1;
            class_tested=1;
        elseif k<=2*Ntest %second class
            tk=t_2;
            class_tested=2;
        else %and third class
            tk=t_3;
            class_tested=3;
        end
        xk=[x_test_2feat(k,:)';1]; %to have a good dimension to multiply by W
        %sigmoid calculation
        zk=Wall_2feat*xk+w_2feat;
        gk=1./(1+exp(-zk));
        [gmax, imax]=max(gk);
        conf_test_2feat(class_tested,imax)=conf_test_2feat(class_tested,imax)+1;
 end
disp('confusion matrix for testing with just 2 features=');
disp(conf_test_2feat);

%error rate for training
error_r_train_2feat=0;
for i=1:C
    for j=1:C
        if i~=j
            error_r_train_2feat=error_r_train_2feat+conf_train_2feat(i,j);
        end
    end
end
error_r_train_2feat=error_r_train_2feat/(Ntrain*C); %error rate is in percentage
disp('error rate for training with 2 features=');
disp(error_r_train_2feat);

%error rate for testing
error_r_test_2feat=0;
for ib=1:C
    for jb=1:C
        if ib~=jb
            error_r_test_2feat=error_r_test_2feat+conf_test_2feat(ib,jb);
        end
    end
end
error_r_test_2feat=error_r_test_2feat/(Ntest*C); %error rate is in percentage
disp('error rate for testing with just 2 features=');
disp(error_r_test_2feat);

%with 1 feature, just petal width left the 4th feature
%training
D=1;
W_3feat=zeros(C,D);%the matrix of the classes with the average attributes
w_3feat=zeros(C,1);%the vector of the classes
Wall_3feat=[W_3feat w_3feat];
MSE_all_3feat=zeros(1,samples);
MSE_grad_all_3feat=zeros(1,samples); %The gradient of each MSE, we need it to calculate the equation (23)

x_train_3feat=x_train(:,4);
x_test_3feat=x_test(:,4);
for i=1:samples
    %for the first class
    MSE=0;
    MSE_grad=0;
    for k=1:size(x_train_3feat,1) %until 90
        if k<=Ntrain %corresponds to the first class
            tk=t_1;
        elseif k<=2*Ntrain %second class
            tk=t_2;
        else %and third class
            tk=t_3;
        end
        xk=[x_train_3feat(k,:)';1]; %to have a good dimension to multiply by W
        %sigmoid calculation
        zk=Wall_3feat*xk+w_3feat;
        gk=1./(1+exp(-zk));
        MSE_grad=MSE_grad+((gk-tk).*(gk).*(1-gk))*xk';
        MSE=MSE+0.5*((gk-tk)'*(gk-tk));
    end
    MSE_grad_all_3feat(i)=norm(MSE_grad);
    MSE_all_3feat(i)=MSE;
    Wall_3feat=Wall_3feat-alpha*MSE_grad;
    
end 

%confusion matrix for training
conf_train_3feat=zeros(C,C); %matrix 3x3
 for k=1:size(x_train_3feat,1) %until 90
        if k<=Ntrain %corresponds to the first class
            tk=t_1;
            class_trained=1;
        elseif k<=2*Ntrain %second class
            tk=t_2;
            class_trained=2;
        else %and third class
            tk=t_3;
            class_trained=3;
        end
        xk=[x_train_3feat(k,:)';1]; %to have a good dimension to multiply by W
        %sigmoid calculation
        zk=Wall_3feat*xk+w_3feat;
        gk=1./(1+exp(-zk));
        [gmax, imax]=max(gk);
        conf_train_3feat(class_trained,imax)=conf_train_3feat(class_trained,imax)+1;
 end
disp('confusion matrix for training with just 1 feature=');
disp(conf_train_3feat);

%confusion matrix for testing
conf_test_3feat=zeros(C,C); %matrix 3x3
 for k=1:size(x_test_3feat,1) %until 90
        if k<=Ntest %corresponds to the first class
            tk=t_1;
            class_tested=1;
        elseif k<=2*Ntest %second class
            tk=t_2;
            class_tested=2;
        else %and third class
            tk=t_3;
            class_tested=3;
        end
        xk=[x_test_3feat(k,:)';1]; %to have a good dimension to multiply by W
        %sigmoid calculation
        zk=Wall_3feat*xk+w_3feat;
        gk=1./(1+exp(-zk));
        [gmax, imax]=max(gk);
        conf_test_3feat(class_tested,imax)=conf_test_3feat(class_tested,imax)+1;
 end
disp('confusion matrix for testing with just 1 feature=');
disp(conf_test_3feat);

%error rate for training
error_r_train_3feat=0;
for i=1:C
    for j=1:C
        if i~=j
            error_r_train_3feat=error_r_train_3feat+conf_train_3feat(i,j);
        end
    end
end
error_r_train_3feat=error_r_train_3feat/(Ntrain*C); %error rate is in percentage
disp('error rate for training with 1 feature=');
disp(error_r_train_3feat);

%error rate for testing
error_r_test_3feat=0;
for ib=1:C
    for jb=1:C
        if ib~=jb
            error_r_test_3feat=error_r_test_3feat+conf_test_3feat(ib,jb);
        end
    end
end
error_r_test_3feat=error_r_test_3feat/(Ntest*C); %error rate is in percentage
disp('error rate for testing with just 1 feature=');
disp(error_r_test_3feat);

%Highlight the linear separability
%As a whole


figure(2);         % Sepal lengths vs. sepal width
subplot(3,2,1); 
hold on;
%for all classes
%scatter(xall(:,1), xall(:, 2), 'filled');
%for class 1
scatter(x1all(:,1), x1all(:, 2), 'filled','blue');
%for class 2
scatter(x2all(:,1), x2all(:, 2), 'filled','red');
%for class 3
scatter(x3all(:,1), x3all(:, 2), 'filled','yellow');
legend('Setosa', 'Versicolour', 'Virginica');
xlabel('Sepal width');
ylabel('Sepal length'); 

subplot(3,2,2); 
         % Petal length vs. petal width
hold on;
scatter(x1all(:, 3), x1all(:, 4), 'filled','blue');
scatter(x2all(:, 3), x2all(:, 4), 'filled','red');
scatter(x3all(:, 3), x3all(:, 4), 'filled','yellow');
legend('Setosa', 'Versicolor', 'Virginica');
xlabel('Petal length');
ylabel('Petal width'); 

subplot(3,2,3); 
         % Petal length vs. sepal length
hold on;
scatter(x1all(:, 3), x1all(:, 1), 'filled','blue');
scatter(x2all(:, 3), x2all(:, 1), 'filled','red');
scatter(x3all(:, 3), x3all(:, 1), 'filled','yellow');
legend('Setosa', 'Versicolor', 'Virginica');
xlabel('Petal width');
ylabel('Sepal length'); 

subplot(3,2,4); 
         % Petal width vs. sepal width
hold on;
scatter(x1all(:, 4), x1all(:, 2), 'filled','blue');
scatter(x2all(:, 4), x2all(:, 2), 'filled','red');
scatter(x3all(:, 4), x3all(:, 2), 'filled','yellow');
legend('Setosa', 'Versicolor', 'Virginica');
xlabel('Petal width');
ylabel('Sepal width'); 

subplot(3,2,5); 
         % Petal width vs. sepal length
hold on;
scatter(x1all(:, 4), x1all(:, 1), 'filled','blue');
scatter(x2all(:, 4), x2all(:, 1), 'filled','red');
scatter(x3all(:, 4), x3all(:, 1), 'filled','yellow');
legend('Setosa', 'Versicolor', 'Virginica');
xlabel('Petal width');
ylabel('Sepal length');

subplot(3,2,6); 
         % Petal length vs. sepal width
hold on;
scatter(x1all(:, 3), x1all(:, 2), 'filled','blue');
scatter(x2all(:, 3), x2all(:, 2), 'filled','red');
scatter(x3all(:, 3), x3all(:, 2), 'filled','yellow');
legend('Setosa', 'Versicolor', 'Virginica');
xlabel('Petal length');
ylabel('Sepal width'); 
