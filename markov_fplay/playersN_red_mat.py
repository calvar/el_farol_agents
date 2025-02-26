#Markov fictitious play as described in -Vanderschraaf and Skyrms, JSTOR 59, 311-348 (2003)-, 
# but modified to allow for the agents to have only partial information about the state of
# the system.
#All agents have the same amount of information
#Case for N players

import numpy as np
from sys import argv
import random
from random import sample
from scipy.special import binom

from datetime import datetime

def bin_to_dec(b_str):
    if len(b_str) > 1:
        return int(b_str[0])*2**(len(b_str)-1) + bin_to_dec(b_str[1:])
    else:
        return int(b_str[0])

def possible_states(N):
    return [np.binary_repr(i,N) for i in range(2**N)]

#Strategies can have value 0 or 1
def av_payoff(N, partial_state, max_attendance):
    Nb = N - len(partial_state)
    k1 = sum([int(i) for i in partial_state])
    mx = int(max_attendance - k1)
    nr = np.array([binom(Nb,u1) for u1 in range(0,mx+1)]).sum()
    pr = nr / 2**(Nb)
    return 2*pr - 1
    
def payoff(N, partial_state, max_attendance):
    po = av_payoff(N, partial_state, max_attendance)
    return [0 if partial_state[i]=="0" else po for i in range(len(partial_state))] 

def softmax_e(e, E0, E1):
    ex0 = e*E0
    ex1 = e*E1
    sm0 = 0
    if ex0 < 300 and ex1 < 300:
        EE0 = np.exp(ex0)
        EE1 = np.exp(ex1)
        tot = EE0+EE1
        sm0 = EE0/tot
    elif ex0 >= 300:
        sm0 = 1
    return (sm0, 1-sm0)

def explore_UCB(C, dist, idx):
    x = np.log(np.sum(dist)) / dist[idx]
    return C*np.sqrt(x)

def verify_matrix(mat):
    for row in mat:
        tot = row.sum()
        assert(1-(1.e-8) < tot and tot < 1+(1.e-8)), "Row total is {0}".format(tot)


# def entropy_rate(freq):
#     tot = 0
#     for i in freq:
#         tot += freq[i]
#     dist = {}
#     for i in freq:
#         dist[i] = freq[i]/tot
#     H = 0
#     for i in range(len(dist)):
#         if dist[i] > 0:
#             H -= dist[i] * np.log2(dist[i])
#     return H/len(dist)


## REDUCED MATRIX APPROACH
# Instead of using a belief transition matrix from system states to system states, use a matrix
# for each agent with 3 states: win (encapsulates states where the agent earns +1), 
# neut (encapsulates states where the agent earns 0), 
# lose (encapsulates states where the agent earns -1). 
#      | win | neut | lose
# win  |  x  |  x   |  x
# neut |  x  |  x   |  x
# lose |  x  |  x   |  x

class Agent:
    def __init__(self, idx, known_idx, N, ini_state, pos_states, Amx):
        self.idx = idx  #Index in the complete set of agents
        self.Ntot = N
        self.state = ini_state
        self.earnings = 0
        self.known_idx = known_idx
        self.mat_pos = self.known_idx.index(self.idx) #Index in the reduced set of known agents
        self.belief_mat = np.zeros((3,3)) # Reduced matrix
        for i in range(3):
            tot = 1
            self.belief_mat[i,0] = np.random.rand()*tot
            tot = 1-self.belief_mat[i,0]
            self.belief_mat[i,1] = np.random.rand()*tot
            self.belief_mat[i,2] = 1-self.belief_mat[i,:2].sum()
        for row in self.belief_mat:
            np.random.shuffle(row)
        #Verify matrix
        verify_matrix(self.belief_mat)
        
        #self.state_earn = {'win': [], 'neut': [], 'lose': []}
        self.payoffs = {'win': 0, 'neut': 0, 'lose': 0}
        n_win = 0
        n_lose = 0
        for state in pos_states:
            po = payoff(self.Ntot, state, Amx)[self.mat_pos]
            if po < 0:
                n_lose += 1
                self.payoffs['lose'] += po
            elif po > 0:
                n_win += 1
                self.payoffs['win'] += po
        self.payoffs['lose'] = 0 if (n_lose==0) else self.payoffs['lose']/n_lose
        self.payoffs['win'] = 0 if (n_win==0) else self.payoffs['win']/n_win
        self.state_hist = [1,1] #Number of no-go (0) and number of go(1). Initialize in 1 to avoid division by 0.

    def expected_payoff(self, curr_state, action):
        known_state = ''.join(curr_state[i] for i in self.known_idx)
        row = self.reduced_idx(known_state)
        Ep = self.payoffs['win'] * self.belief_mat[row,0] \
             + self.payoffs['lose'] * self.belief_mat[row,2]
        return 0 if action=='0' else Ep
    
    def decide_action(self, state, e, C):
        pay0 = self.expected_payoff(state,"0")
        pay1 = self.expected_payoff(state,"1")
        expl0 = explore_UCB(C,self.state_hist,0) 
        expl1 = explore_UCB(C,self.state_hist,1)
            
        if pay0+expl0 > pay1+expl1:
            self.state = "0"
        elif pay0+expl0 < pay1+expl1:
            self.state = "1"
        else:
            self.state = str(random.randint(0,1))
        #sm = softmax_e(e,pay0,pay1)
        #if np.random.rand() < sm[0]:
        #    self.state = "0"
        #else:
        #    self.state = "1"
        self.state_hist[int(self.state)] += 1
        return self.state, pay0, pay1, expl0, expl1
    
    def update_mat(self, curr_state, new_state, old_st_freq, st_freq, lamb):
        known_curr = ''.join(curr_state[i] for i in self.known_idx)
        known_new = ''.join(new_state[i] for i in self.known_idx)
        
        curr_idx = self.reduced_idx(known_curr)  
        new_idx = self.reduced_idx(known_new)  
        for i in range(3):
            for j in range(3):
                if i == curr_idx:
                    self.belief_mat[i,j] *= (old_st_freq[i]+lamb) / (st_freq[i]+lamb)
                    if j == new_idx:
                        self.belief_mat[i,j] += 1. / (st_freq[i]+lamb)
        ##Verify matrix
        #verify_matrix(self.belief_mat)

    def update_earnings(self, Amx, Act_att):
        if self.state == "1":
            if Act_att > Amx:
                self.earnings -= 1
            else:
                self.earnings += 1

    def reduced_idx(self, state):
        po = payoff(self.Ntot, state, Amx)[self.mat_pos]
        idx = 1
        if po > 0:
            idx = 0
        elif po < 0:
            idx = 2
        return idx



if __name__ == '__main__':
    seed = int(argv[1])
    random.seed(seed)
    np.random.seed(seed)

    Niters = 4000 #Number of iterations
    Hptr = 5 #Print interval for entropy rate
    Mptr = 50 #Print interval for matrices
    reset_time = 1000 #Step at which state frequency is reset

    N = 12 #Number of agents
    thresh = 1/4 #Attendance threshold
    b = 12 #Number of bits available to each agent. 
          # By default, the agent has acces to its own previous state.
    assert N >= b
    e = 64 #Inverse temperature
    lamb = 1 #Weight of the generalized succession rule (equal for all in this case)

    C = 0.01 #exploration function amplitude
    
    Amx = np.floor(N*thresh) #Max number of 1s allowed to get payoff = 1
    pos_states = possible_states(b)

    #Initialize----------------------------
    #known_idx includes the self index plus other b-1 indices 
    rand_sample = lambda x: sorted(sample([j for j in range(N) if j!=x],k=b-1)+[x]) #use to obtain random indices
    cycl_sample = lambda x: sorted([(j+1)%N for j in range(x,x+b-1)]+[x]) #use to obtain fixed sequential indices
    disc_sample = lambda x: sorted([j for j in range(b)] if x<b else [x]+[j for j in range(1,b)]) #disconnected. No one but themselves sees the last N-b indices 
    agents = [Agent(i,rand_sample(i),N,str(random.randint(0,1)),pos_states,Amx) for i in range(N)]
    with open("data_p/known_idx_{0:04d}.dat".format(seed),"w") as outf:
        for i in range(len(agents)):
            outf.write("{0}\t{1}\n".format(i,' '.join(str(e) for e in agents[i].known_idx)))

    #for ag in agents:
    #    print(ag.payoffs)

    old_state_freq = [np.zeros(3, dtype=np.uint64) for i in range(N)]
    state_freq = [np.zeros(3, dtype=np.uint64) for i in range(N)]
    #print(pos_states)

    state = ""
    for i in range(N):
         state += str(agents[i].state)
    print("state: {0}".format(state))
    #print("state index: {0}".format(bin_to_dec(state)))

    #for ag in agents:
    #    print(ag.expected_payoff(state,'0'),ag.expected_payoff(state,'1'))

    #for i in range(N):
    #    print(agents[i].known_idx)
    #    print(agents[i].belief_mat)
    #    print(agents[i].expected_payoff(state, pos_states, Amx, '1'))
        

    #Structures to save output
    state_evol = [state]
    #entropy_evol = []
    earnings_evol = [[0] for i in range(N)]
    distr_evol = [[] for i in range(N)]
    matrix_evol = [[] for i in range(N)]
    payexpl = [[] for i in range(N)]
    for i in range(N):
        distr_evol[i].append(state_freq[i])
        matrix_evol[i].append(agents[i].belief_mat.reshape(-1))
        payexpl[i].append([0, 0, 0, 0])

    #Run iterations-------------------------
    now = datetime.now()
    print('Start time', now.time())
    for t in range(Niters):
        #print(t)

        if t == reset_time:
            old_state_freq = [np.zeros(3, dtype=np.uint64) for i in range(N)]
            state_freq = [np.zeros(3, dtype=np.uint64) for i in range(N)]
        for i in range(N):
            j = agents[i].reduced_idx(state)
            state_freq[i][j] += 1

        #update state
        new_state = ""
        for i in range(N):
            act, pay0, pay1, ex0, ex1 = agents[i].decide_action(state,e,C) 
            new_state += act
            payexpl[i].append([pay0, pay1, ex0, ex1])
        #print("state: {0}".format(new_state))
         
        #update belief matrices
        for i in range(N):
            agents[i].update_mat(state, new_state, old_state_freq[i], state_freq[i], lamb)
        
        #update earnings
        Attendance = np.array([int(i) for i in new_state]).sum()
        for i in range(N):
            agents[i].update_earnings(Amx,Attendance)
        
        #Save data
        if t >= reset_time:
            state_evol.append(new_state)
            for i in range(N):
                earnings_evol[i].append(agents[i].earnings)
            #if t%Hptr == 0:
            #    entropy_evol.append(entropy_rate(state_freq))
            if t%Mptr == 0:
                for i in range(N):
                    distr_evol[i].append(state_freq[i]/state_freq[i].sum())
                    matrix_evol[i].append(agents[i].belief_mat.reshape(-1))
        
        
        for i in range(N):
            j = agents[i].reduced_idx(state)
            old_state_freq[i][j] += 1
        state = new_state

    ## #---------------------------------------

    #Output data to files
    with open("data_p/state_evol_{0:04d}.dat".format(seed),"w") as outf:
        for i in range(len(state_evol)):
            outf.write("{0}\t{1}\n".format(i,state_evol[i]))
            
    with open("data_p/earnings_evol_{0:04d}.dat".format(seed),"w") as outf:
        for j in range(len(earnings_evol[0])):
            outf.write("{0}\t".format(j))
            for i in range(N):
                outf.write("{0} ".format(earnings_evol[i][j]))
            outf.write("\n")

    #with open("data_p/entropy_evol_{0:04d}.dat".format(seed), "w") as outf:
    #    for i in range(len(entropy_evol)):
    #        outf.write("{0}\t{1}\n".format(i*Hptr,entropy_evol[i]))

    for i in range(N):
        with open("data_p/distr_evol_{0}_{1:04d}.dat".format(i,seed),"w") as outf:
            for j in range(len(distr_evol[i])):
                vstr = str(distr_evol[i][j]).replace('\n','').lstrip('[').rstrip(']')
                outf.write("{0}\t{1}\n".format(j*Mptr,vstr))
        
        with open("data_p/matrix_evol_{0}_{1:04d}.dat".format(i,seed),"w") as outf:
            for j in range(len(matrix_evol[i])):
                vstr = str(matrix_evol[i][j]).replace('\n','').lstrip('[').rstrip(']')
                outf.write("{0}\t{1}\n".format(j*Mptr,vstr))

        with open("data_p/payexpl_{0}_{1:04d}.dat".format(i,seed),"w") as outf:
            for j in range(len(payexpl[i])):
                vstr = str(list(map(float,payexpl[i][j]))).replace('\n','').replace(',','').lstrip('[').rstrip(']')
                outf.write("{0}\t{1}\n".format(j,vstr))

    now = datetime.now()
    print('Stop time', now.time())