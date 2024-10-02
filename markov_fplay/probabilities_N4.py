#N=4 agents
#2^4 possible states:
# 0000, 0001, 0010, 0011, 0100, 0101, 0110, 0111,
# 1000, 1001, 1010, 1011, 1100, 1101, 1110, 1111.

import numpy as np


if __name__ == '__main__':
    dim = 2**4

    time = 500

    #Read probability distribution
    pdist = []
    with open("data/distr_evol_0030.dat", "r") as mfile:
        for line in mfile:
            v = list(map(float,line.split()))
            assert len(v)-1 == dim
            if v[0] == time:
                pdist = np.array(v[1:])
    print(pdist)

    #Read Transition matrix (Conditional probability)
    trmat = []
    with open("data/matrix_evol_0_0030.dat", "r") as mfile:
        for line in mfile:
            v = list(map(float,line.split()))
            assert len(v)-1 == dim**2
            if v[0] == time:
                trmat = np.array(v[1:]).reshape(dim,dim)
    print(trmat)

    #Compute joint probability matrix
    jdist = (pdist * trmat.T).T
    print(jdist)
