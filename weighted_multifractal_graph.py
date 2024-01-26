import networkx as nx
from statistics import mean, median, stdev
import matplotlib.pyplot as plt
import random
import statsmodels.stats.api as sms
import numpy as np

# UTILITY FUNCTIONS

def calcIdx(M, K):
  idx = np.zeros((K, M**K), dtype=int)

  for i in range(K):
    idx0 = np.zeros(M**(i + 1), dtype=int)

    for j in range(1, M + 1):
        idx0[(j - 1) * M**i : j * M**i] = np.ones(M**i, dtype=int) * j

    idx[i, :] = np.tile(idx0, M**(K - 1 - i))
  return idx - 1


def calcLK(M, K, idx, L):
    # Calculate K-1 th kronecker product of L
    LK = np.zeros(M**K)
    LKcum = np.zeros(M**K)

    for i in range(M**K):
      LK[i] = np.prod(L[idx[:, i]])
      LKcum[i] = np.sum(LK[0:i+1])

    return LK, LKcum


def calcPK(M, K, idx, P):
    # Calculate K-1 th kronecker product of P
    PK = np.zeros((M**K, M**K))

    for i in range(M**K):
        for j in range(M**K):
            PK[i, j] = np.prod(np.diag(P[idx[:, i], idx[:, j]]))

    return PK


def repeated_kronecker_product(matrix, repetitions):
    # Calculate K-1 th kronecker product of P using numpy kron
    result_matrix = matrix
    for i in range(repetitions - 1):
      result_matrix = np.kron(result_matrix, matrix)
    return result_matrix


def discretized_adj(adj_matrix):
    rounded_matrix = np.where(adj_matrix < 64, 2 ** np.round(np.log2(adj_matrix)), 64)
    return rounded_matrix.astype(int)


def WMGpreprocessing(N, M, K, networkIdx, isDirected, isBinary):
    # Convert adjacency matrix to edge vectors
    edgeCnt = 0
    edgeIdx0 = np.zeros((N**2, 3))

    if isDirected:
        # Directed graph
        for u in range(1, N+1):
            for v in range(1, N+1):
                if networkIdx[u-1, v-1] != 0:
                    edgeIdx0[edgeCnt, :] = [u, v, networkIdx[u-1, v-1]]
                    edgeCnt += 1
    else:
        # Undirected graph
        for u in range(1, N+1):
            for v in range(u, N+1):
                if networkIdx[u-1, v-1] != 0:
                    edgeIdx0[edgeCnt, :] = [u, v, networkIdx[u-1, v-1]]
                    edgeCnt += 1

    edgeIdx = edgeIdx0[:edgeCnt, :]

    # Calc iterDecomp coefficients N
    coeffN = np.zeros((M**K, K))
    for i in range(M**K):
        for idx in range(1, K+1):
            coeffN[i, K-idx] = (i // (M**(idx-1))) % M + 1

    # Calc coefficients m(i,j,i0,j0)
    coeffM = np.zeros((M**K, M**K, M, M))

    if isDirected:
        # Directed graph, unsymmetric P, coeffM(i,j,i0,j0)
        for i in range(M**K):
            for j in range(M**K):
                for i0 in range(1, M+1):
                    for j0 in range(1, M+1):
                        for k in range(1, K+1):
                            if coeffN[i, k-1] == i0 and coeffN[j, k-1] == j0:
                                coeffM[i, j, i0-1, j0-1] += 1
    else:
        # Undirected graph, symmetric P, coeffM(i,j,i0,j0), where i0<=j0
        for i in range(M**K):
            for j in range(M**K):
                for i0 in range(1, M+1):
                    for j0 in range(i0, M+1):
                        for k in range(1, K+1):
                            if min(coeffN[i, k-1], coeffN[j, k-1]) == i0 and max(coeffN[i, k-1], coeffN[j, k-1]) == j0:
                                coeffM[i, j, i0-1, j0-1] += 1

    return edgeIdx, edgeCnt, coeffM, coeffN



def varEStep(N, networkIdx, M, K, modelParaPK, modelParaLK, iterMaxE, isDirected, isBinary):
    tau_log_new = np.zeros((N, M**K))
    tau_log = np.zeros((N, M**K))
    lambda_val = np.zeros(N)
    tau = np.tile(np.transpose(modelParaLK), (N, 1))

    for iterE in range(1, iterMaxE + 1):
      print("Iteration - varEStep", iterE)
      for u in range(N):
        for q in range(M**K):
          tau_log_new[u, q] = np.log(modelParaLK[q])
          for v in range(1, N):  # Exclude self-loop
              for l in range(M**K):
                if isDirected and not isBinary:
                    tau_log_new[u, q] += tau[v, l] * 2 * np.log(
                        (modelParaPK[q, l]**networkIdx[u, v]) * (1 - modelParaPK[q, l])
                    )
                if not isDirected and not isBinary:
                    tau_log_new[u, q] += tau[v, l] * np.log(
                        (modelParaPK[q, l]**networkIdx[u, v]) * (1 - modelParaPK[q, l])
                    )
                if isDirected and isBinary:
                    tau_log_new[u, q] += tau[v, l] * 2 * np.log(
                        (modelParaPK[q, l]**networkIdx[u, v]) *
                        (1 - modelParaPK[q, l])**(1 - networkIdx[u, v])
                    )
                if not isDirected and isBinary:
                    tau_log_new[u, q] += tau[v, l] * np.log(
                        (modelParaPK[q, l]**networkIdx[u, v]) *
                        (1 - modelParaPK[q, l])**(1 - networkIdx[u, v])
                    )

        lambda_val[u] = np.max(tau_log_new[u, :])
        tau_log[u, :] = tau_log_new[u, :] - lambda_val[u] * np.ones(M**K)

      tau = np.exp(tau_log)
      tau = tau / np.sum(tau, axis=1, keepdims=True)

    return tau


def varMStep(N, networkIdx, tau, M, K, modelParaP, modelParaPK, iterMaxM, coeffN, coeffM, idx, stepLen,
             isDirected, isBinary, delta_stopping_M, keep_paraL, modelParaL, modelParaLK):
    llhP = np.zeros(iterMaxM)

    # ModelParaL
    if keep_paraL == 0:
        numerator = np.zeros(M)
        for i in range(M):
            for u in range(N):
                for q in range(M**K):
                    numerator[i] += tau[u, q] * np.sum(coeffN[q, :] == (i+1))
        modelParaL = numerator / (N * K)
        modelParaLK = repeated_kronecker_product(modelParaL, K)
        modelParaLK[modelParaLK == 0] = 0.00001  # LK~=0 constraints
        modelParaLK = modelParaLK / np.sum(modelParaLK)

    # ModelParaP
    if not isDirected:
        # Undirected graph
        for iterM in range(1, iterMaxM + 1):
            # Calc gradientP
            gradientP = np.zeros((M, M))
            for i in range(M):
                for j in range(i, M):
                    for u in range(N):
                        for v in range(1, N):  # Exclude self-loop
                            for q in range(M**K):
                                for l in range(M**K):
                                    if not isBinary:
                                        # Undirected, weighted
                                        gradientP[i, j] += tau[u, q] * tau[v, l] * (networkIdx[u, v] - modelParaPK[q, l] / (1 - modelParaPK[q, l])) * coeffM[q, l, i, j]
                                    else:
                                        # Undirected, binary
                                        gradientP[i, j] += tau[u, q] * tau[v, l] * (networkIdx[u, v] - modelParaPK[q, l] * (1 - networkIdx[u, v]) / (1 - modelParaPK[q, l])) * coeffM[q, l, i, j]

                    gradientP[i, j] /= modelParaP[i, j]


            # Step on
            modelParaP += stepLen * gradientP

            # Constraint: [0,1]
            modelParaP[modelParaP <= 0] = 0.01
            modelParaP[modelParaP >= 1] = 0.99

            # Complete modelParaP
            modelParaP = (modelParaP + modelParaP.T) / 2

            modelParaPK = repeated_kronecker_product(modelParaP, K)

            # Calc llhP
            llhP[iterM - 1] = 0
            for u in range(N):
                for v in range(1, N):  # Exclude self-loop
                    for q in range(M**K):
                        for l in range(M**K):
                            if not isBinary:
                                # Undirected, weighted
                                llhP[iterM - 1] += tau[u, q] * tau[v, l] * np.log(modelParaPK[q, l]**networkIdx[u, v] * (1 - modelParaPK[q, l]))
                            else:
                                # Undirected, binary
                                llhP[iterM - 1] += tau[u, q] * tau[v, l] * np.log(modelParaPK[q, l]**networkIdx[u, v] * (1 - modelParaPK[q, l])**(1 - networkIdx[u, v]))

            llhP[iterM - 1] /= 2

            print("Iteration - varMstep ", iterM)

            # Stopping rule
            if iterM > 5:
                delta = (llhP[iterM - 1] - llhP[iterM - 2]) / llhP[iterM - 2]
                if abs(delta) < delta_stopping_M:
                    break

    else:
        # Directed graph
        for iterM in range(1, iterMaxM + 1):
            # Calc gradientP
            gradientP = np.zeros((M, M))
            for i in range(M):
                for j in range(M):
                    for u in range(N):
                        for v in range(1, N):  # Exclude self-loop
                            for q in range(M**K):
                                for l in range(M**K):
                                    if not isBinary:
                                        # Directed, weighted
                                        gradientP[i, j] += tau[u, q] * tau[v, l] * (networkIdx[u, v] - modelParaPK[q, l] / (1 - modelParaPK[q, l])) * coeffM[q, l, i, j]
                                    else:
                                        # Directed, binary
                                        gradientP[i, j] += tau[u, q] * tau[v, l] * (networkIdx[u, v] - modelParaPK[q, l] * (1 - networkIdx[u, v]) / (1 - modelParaPK[q, l])) * coeffM[q, l, i, j]

                    gradientP[i, j] /= modelParaP[i, j]

            # Step on
            modelParaP += stepLen * gradientP

            # Constraint: [0,1]
            modelParaP[modelParaP <= 0] = 0.01
            modelParaP[modelParaP >= 1] = 0.99

            modelParaPK = repeated_kronecker_product(modelParaP, K)


            # Calc llhP
            llhP[iterM - 1] = 0
            for u in range(N):
                for v in range(1, N):  # Exclude self-loop
                    for q in range(M**K):
                        for l in range(M**K):
                            if not isBinary:
                                # Directed, weighted
                                llhP[iterM - 1] += tau[u, q] * tau[v, l] * np.log(modelParaPK[q, l]**networkIdx[u, v] * (1 - modelParaPK[q, l]))
                            else:
                                # Directed, binary
                                llhP[iterM - 1] += tau[u, q] * tau[v, l] * np.log(modelParaPK[q, l]**networkIdx[u, v] * (1 - modelParaPK[q, l])**(1 - networkIdx[u, v]))

            # Stopping rule
            if iterM > 5:
                delta = (llhP[iterM - 1] - llhP[iterM - 2]) / llhP[iterM - 2]
                if abs(delta) < delta_stopping_M:
                    break

    # llh
    llhL = np.sum(tau * np.log(np.tile(modelParaLK, (N, 1))))
    tau_nonzero = tau[tau != 0]
    llhQ = np.sum(tau_nonzero * np.log(tau_nonzero))
    llh = llhL + llhP[iterM - 1] - llhQ

    return modelParaL, modelParaLK, modelParaP, modelParaPK, llhL, llhQ, llhP, llh, iterM


def EMalgorithm(adj_matrix, N, M, K, isDirected, isBinary, keep_ParaL):

    idx = calcIdx(M, K)  # Function call to calculate index matrix

    modelParaP0 = np.array([[0.1, 0.12], [0.12, 0.1]])  # Initial probability matrix for edges
    modelParaL0 = np.ones((M, 1)) * 1 / M  # Initial probabilities for node attributes

    # EM settings
    iterMax = 300
    iterMaxE = 10
    iterMaxM = 100
    stepLen = 5e-7
    delta_stopping = 1e-1
    delta_stopping_M = 1e-3

    # w(r)={0,2,4,8,16,32,64,max}
    networkIdx = discretized_adj(adj_matrix)

    # Call the pre-processing function
    edgeIdx, edgeCnt, coeffM, coeffN = WMGpreprocessing(N, M, K, networkIdx, isDirected, isBinary)

    # Run varEM
    # networkIdx = adj_d #adj_d discretized matrix
    # Main algorithm follows, but it's not fully provided in the given code
    # It involves the reconstruction of the WMGM through a variational EM algorithm
    # The algorithm includes E-step and M-step for updating parameters iteratively
    # The stopping criteria include maximum iterations and convergence checks
    # See the comments in the provided code for details

    iterM = np.zeros(iterMax)
    llh = np.zeros((iterMax, 1))
    llhQ = np.zeros((iterMax, 1))
    llhL = np.zeros((iterMax, 1))
    llhP = np.zeros((iterMax, iterMaxM))
    paraP_all = np.zeros((M, M, iterMax))
    paraL_all = np.zeros((M, iterMax))

    modelParaP = modelParaP0
    modelParaL = modelParaL0
    modelParaPK = repeated_kronecker_product(modelParaP0, K)
    modelParaLK = repeated_kronecker_product(modelParaL0, K)

    for iter in range(iterMax):

        # E-step: update tau
        tau = varEStep(N, networkIdx, M, K, modelParaPK, modelParaLK, iterMaxE, isDirected, isBinary)

        # M-step: update p and l
        (modelParaL, modelParaLK, modelParaP, modelParaPK, llhL[iter, 0], llhQ[iter, 0], llhP[iter, :], llh[iter, 0],
         iterM[iter]) = varMStep(N, networkIdx, tau, M, K, modelParaP, modelParaPK, iterMaxM, coeffN, coeffM, idx, stepLen,
                             isDirected, isBinary, delta_stopping_M, keep_ParaL, modelParaL, modelParaLK)

        paraP_all[:, :, iter - 1] = modelParaP
        paraL_all[:, iter - 1] = modelParaL

        # Print
        print(f'EM iter={iter} finished, llh={llh[iter - 1]}')
        print(modelParaP)
        print(modelParaL)

        # Stopping rules
        if iter > 10:
            delta_llh = llh[iter] - llh[iter - 1]
            if abs(delta_llh) < delta_stopping:
                break
            if delta_llh < 0:
                break

    np.savetxt('EM_py_results/modelParaL.csv', modelParaL, delimiter=';')
    np.savetxt('EM_py_results/modelParaLK.csv', modelParaLK, delimiter=';')
    np.savetxt('EM_py_results/modelParaP.csv', modelParaP, delimiter=';')
    np.savetxt('EM_py_results/modelParaPK.csv', modelParaPK, delimiter=';')
    np.savetxt('EM_py_results/lhL.csv', llhL, delimiter=';')
    np.savetxt('EM_py_results/llhQ.csv', llhQ, delimiter=';')
    np.savetxt('EM_py_results/llhP.csv', llhP, delimiter=';')
    np.savetxt('EM_py_results/llh.csv', llh, delimiter=';')
    np.savetxt('EM_py_results/iterM.csv', iterM, delimiter=';')



def generate_adj(PK, LK, LKcum, N, isDirected, isBinary):
    """
    Generate the adjacency matrix of the weighted multifractal graph given
    - PK k-1-th kronecker product of the probability of generate edges p
    - LK k-1-th kronecker product of the probability of containing a node l
    - cumulative sum of LK
    - N number of nodes in the graph
    - flag isDirected
    - flag isBinary
    """

    adj = np.zeros((N, N))
    class_array = np.zeros(N)

    # Generating nodes
    for i in range(N):
        tmpNode = np.random.rand()
        class_array[i] = np.argmax(tmpNode <= LKcum)

    class_array = class_array.astype(int)

    # Generating links
    rMax = 14
    w = np.arange(rMax + 1)
    Q = len(LK)
    cmf = np.zeros((Q, Q, rMax + 1))

    for q in range(Q):
        for l in range(Q):
            p = PK[q, l]
            cmf[q, l, 0] = 1 - p
            for r in range(rMax):
                cmf[q, l, r + 1] = cmf[q, l, r] + (1 - p) * p**r

    if isDirected:
        # Directed graph, unsymmetric P, UNTESTED!!
        for i in range(N):
            for j in [x for x in range(i - 1)] + [x for x in range(i + 1, N)]:
                tmpLink = np.random.rand()
                if tmpLink > cmf[class_array[i], class_array[j], -1]:
                    adj[i, j] = w[-1]
                else:
                    adj[i, j] = w[np.where(tmpLink <= cmf[class_array[i], class_array[j], :])[0][0]]
    else:
        # Undirected graph, symmetric P
        for i in range(N):
            for j in range(i + 1, N):
                tmpLink = np.random.rand()
                if tmpLink > cmf[class_array[i], class_array[j], -1]:
                    adj[i, j] = w[-1]
                else:
                    adj[i, j] = w[np.where(tmpLink <= cmf[class_array[i], class_array[j], :])[0][0]]
                adj[j, i] = adj[i, j]

    # Convert weighted to binary graph
    if isBinary:
        adj[adj > 0] = 1

    return adj, class_array