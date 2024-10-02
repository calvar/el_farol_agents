#Markov fictitious play as described in -Vanderschraaf and Skyrms, JSTOR 59, 311-348 (2003)-
#Case for 2 players

import numpy as np

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
    if Attendance/2 > thresh:
        return [0 if state[i]=="0" else -1 for i in range(2)] 
    else:
        return [0 if state[i]=="0" else 1 for i in range(2)]
    
    
class Agent:
    def __init__(self, idx, ini_state):
        self.idx = idx
        self.state = ini_state
        sz = 2**2
        self.belief_mat = np.zeros((sz,sz))
        for i in range(sz):
            tot = 1
            self.belief_mat[i,0] = np.random.rand()*tot
            for j in range(1,sz-1):
                tot = 1-self.belief_mat[i,:j].sum()
                self.belief_mat[i,j] = np.random.rand()*tot
            self.belief_mat[i,sz-1] = 1-self.belief_mat[i,:sz-1].sum()

    def expected_payoff(self, curr_state, pos_states, action):
        column = bin_to_dec(curr_state)
        n_states = [st for st in pos_states if st[self.idx]==action]
        Ep = 0
        for s in n_states:
            Ep += payoff(s,0.5)[self.idx] * self.belief_mat[bin_to_dec(s),column]
        return Ep

    def update_mat(self, curr_state, new_state, old_st_freq, st_freq, lamb):
        curr_idx = bin_to_dec(curr_state)
        new_idx = bin_to_dec(new_state)
        for i in range(self.belief_mat.shape[0]):
            for j in range(self.belief_mat.shape[1]):
                if i == curr_idx:
                    self.belief_mat[i,j] *= (old_st_freq[i]+lamb) / (st_freq[i]+lamb)
                    if j == new_idx:
                        self.belief_mat[i,j] += 1. / (st_freq[i]+lamb)
                
    def decide_action(self):
        pay0 = self.expected_payoff(state,pos_states,"0")
        pay1 = self.expected_payoff(state,pos_states,"1")
        if pay0 > pay1:
            return "0"
        else:
            return "1"


if __name__ == '__main__':
    np.random.seed(331)
    # num = 5
    # b_num = np.binary_repr(num) 
    # print(b_num)
    # numn = bin_to_dec(b_num)
    # print(numn)

    Niters = 50
    Nptr = 10
    lamb = 1 #Weight of the generalized succession rule (equal for all in this case)
    
    agents = [Agent(i,np.random.randint(0,2)) for i in range(2)]

    pos_states = possible_states(2)
    old_state_freq = np.zeros(2**2, dtype=np.uint8)
    state_freq = old_state_freq.copy()
    print(pos_states)
    
    state = str(agents[0].state)+str(agents[1].state)
    print("state: {0}".format(state))
    #print("state index: {0}".format(bin_to_dec(state)))

    print(agents[0].belief_mat)
    print(agents[1].belief_mat)



    for t in range(Niters):
        state_freq[bin_to_dec(state)] += 1
        
        new_state = agents[0].decide_action()+agents[1].decide_action()
        
        agents[0].update_mat(state, new_state, old_state_freq, state_freq, lamb)
        agents[1].update_mat(state, new_state, old_state_freq, state_freq, lamb)

        if t%Nptr == 0:
            print("Iter: {0}".format(t))
            print("new state: {0}".format(new_state))
            print(state_freq)
            print(agents[0].belief_mat)
            print(agents[1].belief_mat)
        
        old_state_freq[bin_to_dec(state)] += 1
        state = new_state

    
