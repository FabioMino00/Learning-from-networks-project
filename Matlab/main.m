%% same initialization
modelParaP = modelParaP0;
modelParaL = modelParaL0;
modelParaPK = calcPK(M,K,idx,modelParaP);
[modelParaLK,~] = calcLK(M,K,idx,modelParaL);


%% preprocessing: calc coeffM and coeffN
[~,~,coeffM,coeffN] = WMGpreprocessing(N,M,K,networkIdx,isDirected,isBinary);

%% start EM iteration
iterM = zeros(1,iterMax);
llh = zeros(iterMax,1);
llhQ = zeros(iterMax,1);
llhL = zeros(iterMax,1);
llhP = zeros(iterMax,iterMaxM);
paraP_all = zeros(M,M,iterMax);
paraL_all = zeros(M,iterMax);

for iter = 1:iterMax
    %% E-step: update tau
    [tau] = varEStep(N,networkIdx,M,K,modelParaPK,modelParaLK,iterMaxE,isDirected,isBinary);
    
    %% M-tep: update p and l
    [modelParaL,modelParaLK,modelParaP,modelParaPK,llhL(iter),llhQ(iter),llhP(iter,:),llh(iter),iterM(iter)] = ...
        varMStep(N,networkIdx,tau,M,K,modelParaP,modelParaPK,iterMaxM,coeffN,coeffM,idx,stepLen,...
        isDirected,isBinary,delta_stopping_M,keep_ParaL,modelParaL,modelParaLK);
    paraP_all(:,:,iter) = modelParaP;
    paraL_all(:,iter) = modelParaL;
    
    %% print
    disp(strcat('EM iter=',num2str(iter),' finished, ','llh=',num2str(llh(iter))));
    disp(modelParaP);
    disp(modelParaL);
    
    %% stopping rules
    if(iter>10)
        %delta_llh = (llh(iter) - llh(iter-1))/llh(iter-1);
        delta_llh = llh(iter) - llh(iter-1);
        if(abs(delta_llh)<delta_stopping)
            break;
        end
        if(delta_llh<0) % llh start increasing/fluctuating
            break;
        end
    end
end
