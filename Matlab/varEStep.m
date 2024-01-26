function[tau]=varEStep(N,networkIdx,M,K,modelParaPK,modelParaLK,iterMaxE,isDirected,isBinary)

tau_log_new = zeros(N,M^K);
tau_log = zeros(N,M^K);
lambda = zeros(N,1);
tau = repmat(modelParaLK,N,1);%tau = rand(N,M^K);tau = tau ./ sum(tau,2);
for iterE = 1:iterMaxE % fixed-point iteration
    for u = 1:N
        for q= 1:M^K
            tau_log_new(u,q) = log(modelParaLK(q));
            for v = [1:u-1,u+1:N] % no self lopp
            %for v = 1:N % contains self loops, if exist
                for l =1:M^K
                    %% directed, weighted
                    if(isDirected && ~isBinary)
                        tau_log_new(u,q) = tau_log_new(u,q) + tau(v,l)*2*...
                            log((modelParaPK(q,l)^networkIdx(u,v))*(1-modelParaPK(q,l)));
                    end
                    %% undirected, weighted
                    if(~isDirected && ~isBinary)
                        tau_log_new(u,q) = tau_log_new(u,q) + tau(v,l)*...
                            log((modelParaPK(q,l)^networkIdx(u,v))*(1-modelParaPK(q,l)));
                    end
                    %% directed, binary
                    if(isDirected && isBinary)
                        tau_log_new(u,q) = tau_log_new(u,q) + tau(v,l)*2*...
                            log((modelParaPK(q,l)^networkIdx(u,v))*(1-modelParaPK(q,l))^(1-networkIdx(u,v)));
                    end
                    %% undirected, binary
                    if(~isDirected && isBinary)
                        tau_log_new(u,q) = tau_log_new(u,q) + tau(v,l)*...
                            log((modelParaPK(q,l)^networkIdx(u,v))*(1-modelParaPK(q,l))^(1-networkIdx(u,v)));
                    end
                end
            end
        end
        % considering the normalization and very small (neg) value of
        % tau_log_new would cause all 0 when taking the exp(~)
        % operation, here we set lambda(u) as the minimum value of
        % tau_log_new(u,:).
        lambda(u) = max(tau_log_new(u,:));
        tau_log(u,:) = tau_log_new(u,:) - lambda(u)*ones(1,M^K);
    end
    tau = exp(tau_log);
    %     tau0 = tau;%
    tau = tau ./ sum(tau,2); % exp(lbd_u - 1) is the normalizing constant
    %         llhE(iter,iterE) = calcLLH(N,M,K,tau,modelParaLK,modelParaPK,networkIdx);
end