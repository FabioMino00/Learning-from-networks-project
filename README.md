# Learning-from-networks-project
Repository containing material for the Learning from network project on Comparative Analysis of Natural and Artificial Neural Networks: Insights from Graph Analysis Metrics

## Repository content
- _Matlab_ directory contains Matlab code from the [Repository](https://github.com/ruocheny/Weighted-Multifractal-Graph-Model/tree/main) which we used as guide for our Weighted Multifractal graph python implementation and to generate the k-1 kronecker product of the edge probability p and node probability l, parameters needed to generate the multifractal graph. In particular the file _rattus_norvegicus.m_ must be run to obtain the l and p vector, stored in csv file inside the directory _Matlab/EM_matlab_result_
- _weighted_multifractal_graph.py_, python file containing all the functions to generate p, l and the adjacency matrix of a Weighted Multifractal graph. It is our implementation of the matlab code above but it is 10 times slower.
- _main.py_ python file that if runned start the EM algorithm to obtain p and l vectors to generate the Weighted Multifractal graph. The two output vectors and other parameters are stored in the EM_py_results directory
- _EM_py_results_ directory containing the generation probabilities p and l estimated using file above.
- _rattus.norvegicus_brain_2.graphml_, dataset available at this [link](https://neurodata.io/project/connectomes/) which is a connectome of a rattus brain
- _Learning_from_networks.ipynb_, jupyter notebook containing all the code to calculate the metrics considered for the comparison between generated graphs and real neural network
- _calculated_metrics_ directory containing a file with all the metrics calculated by the notebook above, pay attention that the code appends the last metrics calculation result at the end of the file _metrics.txt_
- _requirements.txt_ containing all the python packages needed to run our code and that can be installed with pip
