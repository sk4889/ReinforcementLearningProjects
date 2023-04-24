import gym
import numpy as np
from gym.spaces import Discrete, Box
import random
import pandas as pd

from matplotlib import pyplot as plt

num_of_episodes = 1
values = {}
policy = {}
gamma = 0.95
render = False
epsilon = 0.1
alpha = 0.1


n_bins = 10
bins = pd.cut([-0.75, 1.51], bins=n_bins, retbins=True)[1][1:-1]

def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]

def get_state_id(state,bins):
    return build_state([to_bin(state[0], bins), to_bin(state[1], bins), to_bin(state[2], bins), 
                           to_bin(state[3], bins), to_bin(state[4], bins), to_bin(state[5], bins), 
                           state[6],state[7]])

def get_action(state,policy,epsilon):
    randomness = np.random.uniform(0,1)
    
    if randomness <= epsilon:
        return random.choice([0,1,2,3])
    else:
        if(state in policy.keys()):
            return policy[state]
        else:
            return random.choice([0,1,2,3])

def generate_episode(env):
#    env.seed(0)
    state = np.around(env.reset(), decimals=1)
    
    trajectory = []

    while True:

        s = get_state_id(state,bins)
        a = get_action(s,policy,epsilon)
        next_state, reward, done, info = env.step(a)
        s_prime = get_state_id(next_state,bins)

        
        if(render):
            env.render()

        trajectory.append((s, a, reward, s_prime))
        
        if done:
            env.close()
            
            if((s,a) in values.keys()):
                Q_s_a = values[(s,a)][-1] 
            else:
                Q_s_a = 0
                
            Q_s_a = Q_s_a + alpha * (reward - Q_s_a)
            
            if((s,a) in values.keys()):
                values[(s,a)].append(Q_s_a)            
            else:
                values[(s,a)] = [Q_s_a]
            
            
            
            some = {}
            for i in range(4):
                if((s,i) in values.keys()):
                    some[values[(s,i)][-1]] = i
            
            policy[s] = some[max(some.keys())]
            
            break
        else:


            if((s,a) in values.keys()):
                Q_s_a = values[(s,a)][-1] 
            else:
                Q_s_a = 0
            
            q_values = {}


            for a_prime in range(4):
                if((s_prime,a_prime) in values.keys()):
                    q_values[values[(s_prime,a_prime)][-1]] = a_prime
            if(len(q_values.keys()) > 0):
                Q_s_prime_a_prime = max(q_values.keys())
            else:
                Q_s_prime_a_prime = 0
                
            Q_s_a = Q_s_a + alpha * (reward + gamma * Q_s_prime_a_prime - Q_s_a)
            
            if((s,a) in values.keys()):
                values[(s,a)].append(Q_s_a)            
            else:
                values[(s,a)] = [Q_s_a]
            
            state = next_state
            
            some = {}
            for i in range(4):
                if((s,i) in values.keys()):
                    some[values[(s,i)][-1]] = i
            
            policy[s] = some[max(some.keys())]

            

    return trajectory

env = gym.make('LunarLander-v2')    
for i in range(50000):
    
    epsilon = 0.1
    
    if(i > 49950):
        epsilon = 0
        print(i)
        render = True
    
    trajectory = generate_episode(env)
    
    
# Plotting Every Visit Values
#plt.title("Monte Carlo Every Visit value estimation")
#plt.ylabel("State Values")
#plt.xlabel("Number Of Occurances")
#for (state,action) in list(values.keys())[:10]:
#    plt.plot(range(1, len(values[(state,action)]) + 1), values[(state,action)], label='State - ' + str((state,action)))
#plt.legend(loc='bottom right')
#plt.show()


    