% Angel Villa 

clc; clear;

train_file = "breast-cancer-wisconsin.csv";

% importing data
train_data = importdata(train_file);
x = train_data(:,1:end-1);
y = train_data(:,end);

N = size(x,1);
D = size(x,2);

% K: number of clusters
% Nit: number of iterations
K = 2;
Nit = 10;

% normalizing data
m = zeros(D,1);
dev = zeros(D,1);

for i=1:D
    m(i) = mean(x(:,i));
    dev(i) = std(x(:,i));
    
    for j=1:N
        x(j,i) = (x(j,i) - m(i))/dev(i);
    end
end

% Main loop, calculates cluster accuracy for each Nit and k
%
% to do: reduce computations
cluster_acc = zeros(Nit,K);
for it=1:Nit
    for k=1:K
        C = cluster(x,k,it);  
        cluster_acc(it,k) = cluster_accuracy(C,x,y,k);
    end
end

% Returns a matrix containing n=1:N and the cluster to which x_n belongs
function c = cluster(x,k,Nit)
    N = size(x,1);
    D = size(x,2);
    c = [(1:N)' zeros(N,1)];
    m = zeros(k,D);
    dist = zeros(k,1);
    
    x_temp = x;
    N_temp = size(x_temp,1);
    
    for i=1:N
        c(i,2) = randi(k);
    end
    
    for it=1:Nit
        for i=1:N
            m(c(i,2),:) = m(c(i,2),:) + x(i,:);
        end
        for j=1:k
            m(j,:) = m(j,:)/sum(c(:,2) == j);
        end
        for i=1:N
            for j=1:k
                dist(j) = norm(x(i,:) - m(j,:));
            end
            [~, arg] = min(dist);
            c(i,2) = arg;
        end
        
        m = zeros(k,D);
    end
end

% Uses the cluster matrix to return clustering accuracy
function acc = cluster_accuracy(C,x,y,k)
    N = size(x,1);
    clusters = [(1:k)' zeros(k,10)];
    counts = [(1:k)' zeros(k,1)];
    for i=1:N
        clusters(C(i,2),y(i) + 2) = clusters(C(i,2),y(i) + 2) + 1;
    end
    
    for i=1:k
        [max_count,~] = max(clusters(i,:));
        counts(i,2) = sum(C(:,2) == i) - max_count;
    end
    
    acc = 1/N*(N - sum(counts(:,2)));
end

