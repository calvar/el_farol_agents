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

#def entropy_rate(freq):
#    tot = freq.sum()
#    dist = freq.copy()/tot
#    H = 0
#    for i in range(dist.shape[0]):
#        if dist[i] > 0:
#            H -= dist[i] * np.log2(dist[i])
#    return H/dist.shape[0]




class Agent:
    def __init__(self, idx, known_idx, ini_state):
        self.idx = idx           #Index in the complete set of agents
        self.state = ini_state
        self.earnings = 0
        self.known_idx = known_idx
        self.mat_pos = self.known_idx.index(self.idx) #Index in the reduced set of known agents
        sz = 2**len(known_idx)
        self.belief_mat = np.zeros((sz,sz))
        for i in range(sz):
            tot = 1
            self.belief_mat[i,0] = np.random.rand()*tot
            for j in range(1,sz-1):
                tot = 1-self.belief_mat[i,:j].sum()
                self.belief_mat[i,j] = np.random.rand()*tot
            self.belief_mat[i,sz-1] = 1-self.belief_mat[i,:sz-1].sum()
        for row in self.belief_mat:
            np.random.shuffle(row)

    def expected_payoff(self, curr_state, pos_states, Amx, action):
        known_state = ''.join(curr_state[i] for i in self.known_idx)
        row = bin_to_dec(known_state)
        n_states = [st for st in pos_states if st[self.mat_pos]==action]
        Ep = 0
        for s in n_states:
            #print(s,payoff(N,s,Amx),self.mat_pos,bin_to_dec(s),row)
            Ep += payoff(N, s, Amx)[self.mat_pos] * self.belief_mat[row,bin_to_dec(s)]
        return Ep

    def decide_action(self, state, pos_states, Amx, e):
        pay0 = self.expected_payoff(state,pos_states,Amx,"0")
        pay1 = self.expected_payoff(state,pos_states,Amx,"1")
        if pay0 > pay1:
            self.state = "0"
        elif pay0 < pay1:
            self.state = "1"
        else:
            self.state = random.randint(0,1)
        #sm = softmax_e(e,pay0,pay1)
        #if np.random.rand() < sm[0]:
        #    self.state = "0"
        #else:
        #    self.state = "1"
        return self.state

    def update_mat(self, curr_state, new_state, old_st_freq, st_freq, lamb):
        known_curr = ''.join(curr_state[i] for i in self.known_idx)
        known_new = ''.join(new_state[i] for i in self.known_idx)
        curr_idx = bin_to_dec(known_curr)
        new_idx = bin_to_dec(known_new)
        for i in range(self.belief_mat.shape[0]):
            for j in range(self.belief_mat.shape[1]):
                if i == curr_idx:
                    self.belief_mat[i,j] *= (old_st_freq[i]+lamb) / (st_freq[i]+lamb)
                    if j == new_idx:
                        self.belief_mat[i,j] += 1. / (st_freq[i]+lamb)
        ##Verify matrix
        #for row in self.belief_mat:
        #    tot = row.sum()
        #   assert(1-(1.e-8) < tot and tot < 1+(1.e-8)), "Row total is {0}".format(tot)

    def update_earnings(self, Amx, Act_att):
        if self.state == "1":
            if Act_att > Amx:
                self.earnings -= 1
            else:
                self.earnings += 1




if __name__ == '__main__':
    seed = int(argv[1])
    random.seed(seed)
    np.random.seed(seed)

    Niters = 2000 #Number of iterations
    Hptr = 5 #Print interval for entropy rate
    Mptr = 50 #Print interval for matrices
    reset_time = 0 #Step at which state frequency is reset

    N = 4 #Number of agents
    thresh = 1/2 #Attendance threshold
    b = 4 #Number of bits available to each agent. 
          # By default, the agent has acces to its own previous state.
    assert N >= b
    e = 64 #Inverse temperature
    lamb = 1 #Weight of the generalized succession rule (equal for all in this case)
    
    Amx = np.floor(N*thresh) #Max number of 1s allowed to get payoff = 1

    #Initialize----------------------------
    #known_idx includes the self index plus other b-1 indices 
    rand_sample = lambda x: sorted(sample([j for j in range(N) if j!=x],k=b-1)+[x]) #use to obtain random indices
    cycl_sample = lambda x: sorted([(j+1)%N for j in range(x,x+b-1)]+[x]) #use to obtain fixed sequential indices
    disc_sample = lambda x: sorted([j for j in range(b)] if x<b else [x]+[j for j in range(1,b)]) #disconnected. No one but themselves sees the last N-b indices 
    agents = [Agent(i, disc_sample(i), str(random.randint(0,1))) for i in range(N)]
    with open("data_p/known_idx_{0:04d}.dat".format(seed),"w") as outf:
        for i in range(len(agents)):
            outf.write("{0}\t{1}\n".format(i,' '.join(str(e) for e in agents[i].known_idx)))

    pos_states = possible_states(b)
    old_state_freq = [np.zeros(2**(b), dtype=np.uint64) for i in range(N)]
    state_freq = [np.zeros(2**(b), dtype=np.uint64) for i in range(N)]
    #print(pos_states)

    state = ""
    for i in range(N):
        state += str(agents[i].state)
    print("state: {0}".format(state))
    #print("state index: {0}".format(bin_to_dec(state)))

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
    for i in range(N):
        distr_evol[i].append(state_freq[i])
        matrix_evol[i].append(agents[i].belief_mat.reshape(-1))

    #Run iterations-------------------------
    for t in range(Niters):
        #print(t)

        #if t == reset_time:
        #    old_state_freq = [np.zeros(2**(N-b), dtype=np.uint64) for i in range(N)]
        #    state_freq = [np.zeros(2**(N-b), dtype=np.uint64) for i in range(N)]
        for i in range(N):
            state_freq[i][bin_to_dec(''.join(state[j] for j in agents[i].known_idx))] += 1

        #update state
        new_state = ""
        for i in range(N):
            new_state += agents[i].decide_action(state,pos_states,Amx,e)
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
            old_state_freq[i][bin_to_dec(''.join(state[j] for j in agents[i].known_idx))] += 1
        state = new_state

    #---------------------------------------

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
    