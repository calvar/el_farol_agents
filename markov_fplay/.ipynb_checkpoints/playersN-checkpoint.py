#Markov fictitious play as described in -Vanderschraaf and Skyrms, JSTOR 59, 311-348 (2003)-
#Case for N players

import numpy as np
from sys import argv
    

def bin_to_dec(b_str):
    if len(b_str) > 1:
        return int(b_str[0])*2**(len(b_str)-1) + bin_to_dec(b_str[1:])
    else:
        return int(b_str[0])

def possible_states(N):
    return [np.binary_repr(i,N) for i in range(2**N)]
    
#Strategies can have value 0 or 1
def payoff(state, thresh):
    Attendance = np.array([int(i) for i in state]).sum()
    if Attendance/len(state) > thresh:
        return [0 if state[i]=="0" else -1 for i in range(len(state))] 
    else:
        return [0 if state[i]=="0" else 1 for i in range(len(state))]

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

def entropy_rate(freq):
    tot = freq.sum()
    dist = freq.copy()/tot
    H = 0
    for i in range(dist.shape[0]):
        if dist[i] > 0:
            H -= dist[i] * np.log2(dist[i])
    return H/dist.shape[0]
        
    
class Agent:
    def __init__(self, idx, N, ini_state):
        self.idx = idx
        self.state = ini_state
        self.earnings = 0
        sz = 2**N
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

    def expected_payoff(self, curr_state, pos_states, thresh, action):
        row = bin_to_dec(curr_state)
        n_states = [st for st in pos_states if st[self.idx]==action]
        Ep = 0
        for s in n_states:
            Ep += payoff(s,thresh)[self.idx] * self.belief_mat[row,bin_to_dec(s)]
        return Ep

    def decide_action(self, state, pos_states, thresh, e):
        pay0 = self.expected_payoff(state,pos_states,thresh,"0")
        pay1 = self.expected_payoff(state,pos_states,thresh,"1")
        if pay0 > pay1:
            self.state = "0"
        else:
            self.state = "1"
        #sm = softmax_e(e,pay0,pay1)
        #if np.random.rand() < sm[0]:
        #    self.state = "0"
        #else:
        #    self.state = "1"
        return self.state
    
    def update_mat(self, curr_state, new_state, old_st_freq, st_freq, lamb):
        curr_idx = bin_to_dec(curr_state)
        new_idx = bin_to_dec(new_state)
        for i in range(self.belief_mat.shape[0]):
            for j in range(self.belief_mat.shape[1]):
                if i == curr_idx:
                    self.belief_mat[i,j] *= (old_st_freq[i]+lamb) / (st_freq[i]+lamb)
                    if j == new_idx:
                        self.belief_mat[i,j] += 1. / (st_freq[i]+lamb)
        ##Verify matrix
        #for row in self.belief_mat:
        #    tot = row.sum()
        #    assert(1-(1.e-8) < tot and tot < 1+(1.e-8)), "Row total is {0}".format(tot)

    def update_earnings(self, thresh, Attendance_th):
        if self.state == "1":
            if Attendance_th > thresh:
                self.earnings -= 1
            else:
                self.earnings += 1
        

if __name__ == '__main__':
    seed = int(argv[1])
    
    np.random.seed(seed)
    # num = 5
    # b_num = np.binary_repr(num) 
    # print(b_num)
    # numn = bin_to_dec(b_num)
    # print(numn)
    
    Niters = 2000 #Number of iterations
    Hptr = 5 #Print interval for entropy rate
    Mptr = 50 #Print interval for matrices
    reset_time = 1500 #Step at which state frequency is reset

    N = 4 #Number of agents
    thresh = 0.5 #Attendance threshold
    e = 64 #Inverse temperature
    lamb = 1 #Weight of the generalized succession rule (equal for all in this case)

    #Initialize----------------------------
    agents = [Agent(i,N,str(np.random.randint(0,2))) for i in range(N)]

    pos_states = possible_states(N)
    old_state_freq = np.zeros(2**N, dtype=np.uint64)
    state_freq = np.zeros(2**N, dtype=np.uint64)
    #print(pos_states)
    
    state = ""
    for i in range(N):
        state += str(agents[i].state)
    print("state: {0}".format(state))
    #print("state index: {0}".format(bin_to_dec(state)))

    #for i in range(N):
    #    print(agents[i].belief_mat)

    #Structures to save output
    state_evol = [state]
    distr_evol = [state_freq]
    entropy_evol = []
    earnings_evol = [[0] for i in range(N)]
    matrix_evol = [[] for i in range(N)]
    for i in range(N):
        matrix_evol[i].append(agents[i].belief_mat.reshape(-1))
        
    #Run iterations-------------------------
    for t in range(Niters):
        #print(t)

        if t == reset_time:
            old_state_freq = np.zeros(2**N, dtype=np.uint64)
            state_freq = np.zeros(2**N, dtype=np.uint64)
        state_freq[bin_to_dec(state)] += 1

        #print(old_state_freq)
        #print(state_freq)
        #for i in range(len(state_freq)):
        #    if np.abs(state_freq[i]-old_state_freq[i]) > 1:
        #        print("BEFORE: Element {0} of state freq has a gap of {1}".format(state_freq[i]-old_state_freq[i]))
        
        #update state
        new_state = ""
        for i in range(N):
            new_state += agents[i].decide_action(state,pos_states,thresh,e)
        
        #update belief matrices
        for i in range(N):
            agents[i].update_mat(state, new_state, old_state_freq, state_freq, lamb)
        
        #update earnings
        Attendance_th = np.array([int(i) for i in new_state]).sum() / N
        for i in range(N):
            agents[i].update_earnings(thresh,Attendance_th)
            
        #Save data
        if t >= reset_time:
            state_evol.append(new_state)
            for i in range(N):
                earnings_evol[i].append(agents[i].earnings)
            if t%Hptr == 0:
                entropy_evol.append(entropy_rate(state_freq))
            if t%Mptr == 0:
                distr_evol.append(state_freq/state_freq.sum())
                for i in range(N):
                    matrix_evol[i].append(agents[i].belief_mat.reshape(-1))
        
        old_state_freq[bin_to_dec(state)] += 1
        state = new_state

    #---------------------------------------

    #Output data to files
    with open("data/state_evol_{0:04d}.dat".format(seed),"w") as outf:
        for i in range(len(state_evol)):
            outf.write("{0}\t{1}\n".format(i,state_evol[i]))

    with open("data/distr_evol_{0:04d}.dat".format(seed),"w") as outf:
        for i in range(len(distr_evol)):
            vstr = str(distr_evol[i]).replace('\n','').lstrip('[').rstrip(']')
            outf.write("{0}\t{1}\n".format(i*Mptr,vstr))
            
    with open("data/earnings_evol_{0:04d}.dat".format(seed),"w") as outf:
        for j in range(len(earnings_evol[0])):
            outf.write("{0}\t".format(j))
            for i in range(N):
                outf.write("{0} ".format(earnings_evol[i][j]))
            outf.write("\n")

    with open("data/entropy_evol_{0:04d}.dat".format(seed), "w") as outf:
        for i in range(len(entropy_evol)):
            outf.write("{0}\t{1}\n".format(i*Hptr,entropy_evol[i]))
            
    for i in range(N):
        with open("data/matrix_evol_{0}_{1:04d}.dat".format(i,seed),"w") as outf:
            for j in range(len(matrix_evol[i])):
                vstr = str(matrix_evol[i][j]).replace('\n','').lstrip('[').rstrip(']')
                outf.write("{0}\t{1}\n".format(j*Mptr,vstr))

    
