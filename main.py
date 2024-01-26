import networkx as nx
from statistics import mean, median, stdev
import matplotlib.pyplot as plt
import random
import statsmodels.stats.api as sms
import numpy as np
import scipy.sparse

from weighted_multifractal_graph import *


def main():
    # Load the GraphML file
    G = (nx.read_graphml('rattus.norvegicus_brain_2.graphml').to_undirected())

    # Print nodes info
    print(G)

    N = G.number_of_nodes()

    # Reference graph adjacency matrix
    adj_matrix = nx.adjacency_matrix(G).toarray()

    # Initialization parameters
    M = 2  # Number of rectangle side divisions per iteration
    K = 3  # Number of iterations
    isDirected = 0  # Flag for directed graph
    isBinary = 0  # Flag for binary graph
    keep_ParaL = 0  # Flag to keep parameters

    EMalgorithm(adj_matrix, N, M, K, isDirected, isBinary, keep_ParaL)


if __name__ == "__main__":
    main()
