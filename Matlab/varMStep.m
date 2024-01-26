function[modelParaL,modelParaLK,modelParaP,modelParaPK,llhL,llhQ,llhP,llh,iterM]=...
    varMStep(N,networkIdx,tau,M,K,modelParaP,modelParaPK,iterMaxM,coeffN,coeffM,idx,stepLen,...
    isDirected,isBinary,delta_stopping_M,keep_paraL,modelParaL,modelParaLK)

llhP = zeros(1,iterMaxM);
%% modelParaL
if keep_paraL==0
    numerator = zeros(1,M);
    for i=1:M
        for u=1:N
            for q=1:M^K
                numerator(i) = numerator(i) + tau(u,q) * sum(coeffN(q,:)==i);
            end
        end
    end
    modelParaL = numerator./(N*K);
    [modelParaLK,~] = calcLK(M,K,idx,modelParaL);
    for q=1:M^K
        if(modelParaLK(q)==0)
            modelParaLK(q)=0.00001;% LK~=0 constraints
        end
    end
    modelParaLK = modelParaLK/sum(modelParaLK);
end

%% modelParaP
if(~isDirected)
    %% undirected graph
    for iterM = 1:iterMaxM
        % calc gradientP
        gradientP = zeros(M,M);
        for i=1:M
            for j=i:M%%
                for u=1:N
                    for v=[1:u-1,u+1:N]
                        %for v=1:N
                        for q=1:M^K
                            for l=1:M^K
                                if(~isBinary)
                                    %% undirected, weighted
                                    gradientP(i,j) = gradientP(i,j) + tau(u,q)*tau(v,l)*...
                                        (networkIdx(u,v)-modelParaPK(q,l)/(1-modelParaPK(q,l)))*coeffM(q,l,i,j);
                                else
                                    %% undirected,binary
                                    gradientP(i,j) = gradientP(i,j) + tau(u,q)*tau(v,l)*...
                                        (networkIdx(u,v)-modelParaPK(q,l)*(1-networkIdx(u,v))/(1-modelParaPK(q,l)))*coeffM(q,l,i,j);
                                end
                            end
                        end
                    end
                end
                gradientP(i,j) = gradientP(i,j)/modelParaP(i,j);
            end
        end
        % step on
        modelParaP = modelParaP + stepLen .* gradientP;% update {p}
        flag = 0;
        % constraint: [0,1]
        for i0=1:M
            for j0=i0:M
                if(modelParaP(i0,j0)<=0 )
                    modelParaP(i0,j0) = 0.01;
                    disp(strcat('p<0 at iterM=',num2str(iterM)));
                    flag = 1;
                end
                if(modelParaP(i0,j0)>=1)
                    modelParaP(i0,j0) = 0.99;
                    disp(strcat('p>1 at iterM=',num2str(iterM)));
                    flag = 1;
                end
            end
        end
        % complete modelParaP
        for i0=1:M
            for j0=1:i0-1
                modelParaP(i0,j0) = modelParaP(j0,i0);
            end
        end
        modelParaPK = calcPK(M,K,idx,modelParaP);
        % calc llhP
        llhP(iterM) = 0;
        for u=1:N
            %for v=1:N
            for v=[1:u-1,u+1:N]
                for q=1:M^K
                    for l=1:M^K
                        if(~isBinary)
                            %% undirected, weighted
                            llhP(iterM) = llhP(iterM) + tau(u,q)*tau(v,l)*...
                                log(modelParaPK(q,l)^networkIdx(u,v)*(1-modelParaPK(q,l)));
                        else
                            %% undirected,binary
                            llhP(iterM) = llhP(iterM) + tau(u,q)*tau(v,l)*...
                                log(modelParaPK(q,l)^networkIdx(u,v)*(1-modelParaPK(q,l))^(1-networkIdx(u,v)));
                        end
                    end
                end
            end
        end
        llhP(iterM) = llhP(iterM)/2;
        
        % stopping rule
        if(flag == 1)
            break;
        end
        if(iterM>5)
            delta = (llhP(iterM) - llhP(iterM-1))/llhP(iterM);
            if(abs(delta)<delta_stopping_M)
                break;
            end
        end
    end
    
    
else
    %% directed graph
    for iterM = 1:iterMaxM
        % calc gradientP
        gradientP = zeros(M,M);
        for i=1:M
            for j=1:M
                for u=1:N
                    for v=[1:u-1,u+1:N]
                        %for v=1:N
                        for q=1:M^K
                            for l=1:M^K
                                if(~isBinary)
                                    %% directed, weighted
                                    gradientP(i,j) = gradientP(i,j) + tau(u,q)*tau(v,l)*...
                                        (networkIdx(u,v)-modelParaPK(q,l)/(1-modelParaPK(q,l)))*coeffM(q,l,i,j);
                                else
                                    %% directed,binary
                                    gradientP(i,j) = gradientP(i,j) + tau(u,q)*tau(v,l)*...
                                        (networkIdx(u,v)-modelParaPK(q,l)*(1-networkIdx(u,v))/(1-modelParaPK(q,l)))*coeffM(q,l,i,j);
                                end
                            end
                        end
                    end
                end
                gradientP(i,j) = gradientP(i,j)/modelParaP(i,j);
            end
        end
        % step on
        modelParaP = modelParaP + stepLen .* gradientP;% update {p}
        flag = 0;
        % constraint: [0,1]
        for i0=1:M
            for j0=1:M
                if(modelParaP(i0,j0)<=0 )
                    modelParaP(i0,j0) = 0.01;
                    disp(strcat('p<0 at iterM=',num2str(iterM)));
                    flag = 1;
                end
                if(modelParaP(i0,j0)>=1)
                    modelParaP(i0,j0) = 0.99;
                    disp(strcat('p>1 at iterM=',num2str(iterM)));
                    flag = 1;
                end
            end
        end
        
        modelParaPK = calcPK(M,K,idx,modelParaP);
        % calc llhP
        llhP(iterM) = 0;
        for u=1:N
            for v=[1:u-1,u+1:N]
                %for v=1:N
                for q=1:M^K
                    for l=1:M^K
                        if(~isBinary)
                            %% undirected, weighted
                            llhP(iterM) = llhP(iterM) + tau(u,q)*tau(v,l)*...
                                log(modelParaPK(q,l)^networkIdx(u,v)*(1-modelParaPK(q,l)));
                        else
                            %% undirected,binary
                            llhP(iterM) = llhP(iterM) + tau(u,q)*tau(v,l)*...
                                log(modelParaPK(q,l)^networkIdx(u,v)*(1-modelParaPK(q,l))^(1-networkIdx(u,v)));
                        end
                    end
                end
            end
            %         llhP(iterM) = llhP(iterM)/2;
            
            % stopping rule
            if(flag == 1)
                break;
            end
            if(iterM>5)
                delta = (llhP(iterM) - llhP(iterM-1))/llhP(iterM-1);
                if(abs(delta)<delta_stopping_M)
                    break;
                end
            end
        end
    end
end

%% llh
llhL = sum(sum(tau.*log(repmat(modelParaLK,N,1))));
tau_nonzero = tau(tau~=0);
llhQ = sum(tau_nonzero.*log(tau_nonzero));
llh = llhL + llhP(iterM) - llhQ;
