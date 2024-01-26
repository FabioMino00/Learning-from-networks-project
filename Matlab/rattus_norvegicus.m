clear
close all

%% load adjacency matrix rattus graph (undirected approximation)
load('rattus_adj_matrix.mat');
adj = adj_matrix_und;
N = size(adj, 1);

%% discretize and weight distribution analysis
adj_vec = reshape(adj,1,N^2);
adj_hist = hist(adj_vec,0:1:max(adj_vec));
adj_d = zeros(N,N);
wbins = [0,2,4,8,16,32,64,max(adj_vec)];
for u=1:N
    for v=1:N
        adj_d(u,v) = find(adj(u,v)<=wbins,1)-1;
    end
end

adj_vec = reshape(adj_d,1,N^2);
xbins = 0:1:max(adj_vec);
adj_hist = hist(adj_vec,xbins);
adj_hist_norm = adj_hist/sum(adj_hist);

%% start estimation
M = 2;
K = 3;
[idx] = calcIdx(M,K);

isDirected = 0;
isBinary = 0;
keep_ParaL = 0;

%% EM settings and initialization
iterMax = 300;
iterMaxE = 10;
iterMaxM = 100;
stepLen = 5e-7;
delta_stopping = 1e-1;
delta_stopping_M = 1e-3;

modelParaP0 = [0.1,0.12;0.12,0.1];
modelParaL0 = ones(M,1)*1/M;

%% run varEM
networkIdx = adj_d;
main;
save('EM_matlab_result\result_rattus.mat');

%% GENERATE OUTPUTS
writematrix(modelParaPK,'EM_matlab_result\modelParaPK.csv')
writematrix(modelParaLK,'EM_matlab_result\modelParaLK.csv')