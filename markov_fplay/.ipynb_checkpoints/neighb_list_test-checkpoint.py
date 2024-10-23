import numpy as np
from random import sample

N = 6
b = 2

rand_sample = lambda x: sorted(sample([j for j in range(N) if j!=x],k=b-1)+[x])
cycl_sample = lambda x: sorted([(j+1)%N for j in range(x,x+b-1)]+[x])
disc_sample = lambda x: sorted([j for j in range(b)] if x<b else [x]+[j for j in range(1,b)])

print("random:", [rand_sample(i) for i in range(N)] )
print("cyclic:", [cycl_sample(i) for i in range(N)] )
print("disconnected:", [disc_sample(i) for i in range(N)] )