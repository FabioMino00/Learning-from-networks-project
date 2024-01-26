function [edgeIdx,edgeCnt,coeffM,coeffN] = WMGpreprocessing(N,M,K,networkIdx,isDirected,isBinary)

%% convert adjacency matrix as edge vectors
edgeCnt = 0;
edgeIdx0 = zeros(N^2,3); % edgeIdx(e,:)=[node u, node v, weight r]
if(isDirected)
    % directed graph
    for u=1:N
        for v=1:N
            if(networkIdx(u,v)~=0)
                edgeIdx0(edgeCnt+1,:) = [u,v,networkIdx(u,v)];
                edgeCnt  = edgeCnt + 1;
            end
        end
    end
else
    % undirected graph
    for u=1:N
        for v = u:N
            %     for v=1:N
            if(networkIdx(u,v)~=0)
                edgeIdx0(edgeCnt+1,:) = [u,v,networkIdx(u,v)];
                edgeCnt  = edgeCnt + 1;
            end
        end
    end
end
edgeIdx = edgeIdx0(1:edgeCnt,:);

%% calc iterDecomp coefficients N
coeffN = zeros(M^K,K);
for i=1:M^K
    %     coeffN(i,:)=iterDecomp(i,M,K);
    for idx = 1:K
        coeffN(i,K+1-idx) = mod(floor((i-1)/(M^(idx-1))),M) + 1;
    end
end

% % %% decomp node index
% % categoryIdx = zeros(M^K,M^K,2,K);
% % for i=1:M^K
% %     for j=1:M^K
% %         % %                 categoryIdx(i,j,1,:) = iterDecomp(i,M,K);
% %         % %                 categoryIdx(i,j,2,:) = iterDecomp(j,M,K);
% %         % %         ik = iterDecomp(i,M,K);
% %         % %         jk = iterDecomp(j,M,K);
% %         ik = coeffN(i,:);
% %         jk = coeffN(j,:);
% %         categoryIdx(i,j,1,:) = min(ik,jk);
% %         categoryIdx(i,j,2,:) = max(ik,jk);
% %     end
% % end

%% calc coeffcients m(i,j,i0,j0)
coeffM = zeros(M^K,M^K,M,M);
if(isDirected)
    %% directed graph, unsymmetric P, coeffM(i,j,i0,j0)
    for i=1:M^K
        for j=1:M^K
            for i0=1:M
                for j0=1:M
                    for k=1:K
                        if(coeffN(i,k)==i0 && coeffN(j,k)==j0)
                            coeffM(i,j,i0,j0) = coeffM(i,j,i0,j0) + 1;
                        end
                    end
                end
            end
        end
    end
else
    %% undirected graph, symmetric P, coeffM(i,j,i0,j0), where i0<=j0
    for i=1:M^K
        for j=1:M^K
            for i0=1:M
                for j0=i0:M
                    for k=1:K
                        if(min(coeffN(i,k),coeffN(j,k))==i0 && max(coeffN(i,k),coeffN(j,k))==j0)
                            coeffM(i,j,i0,j0) = coeffM(i,j,i0,j0) + 1;
                        end
                    end
                end
            end
        end
    end
end
