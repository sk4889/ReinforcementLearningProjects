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
    state = env.reset()
    
    trajectory = []

    while True:

        state_id = get_state_id(state,bins)
        action = get_action(state_id,policy,epsilon)
        next_state, reward, done, info = env.step(action)
        next_state_id = get_state_id(next_state,bins)
        
        if(render):
            env.render()

        trajectory.append((state_id, action, reward, next_state_id))
        
        state = next_state
        
        if done:
            env.close()
            break

    return trajectory

def calculate_values(single_visit, orig_trajectory, values):
    
    trajectory = orig_trajectory.copy()

    if len(trajectory) > 1:
        trajectory.reverse()

    G = 0
    visited_states = []

    for state, action, reward, next_state in trajectory:

        G = gamma * G + reward

        if single_visit and (state,action) in visited_states:
            continue

        visited_states.append((state,action))
        
        if((state,action) in values.keys()):
            previous_value = values[(state,action)][-1]
            count = len(values[(state,action)]) + 1
        else:
            previous_value = 0                
            count = 1                
            
        value = previous_value + ((G - previous_value) / count)
#        print((state,action) , " -> " , value)
        
        if((state,action) in values.keys()):
            values[(state,action)].append(value)
        else:
            values[(state,action)] = [value]
            
#        print(values)
        some = {}
        for i in range(4):
            if((state,i) in values.keys()):
                some[values[(state,i)][-1]] = i
        
        policy[state] = some[max(some.keys())]
        

env = gym.make('LunarLander-v2')
    
for i in range(5000):
    print(i)
    epsilon = 0.1
    
    if(i > 4950):
        epsilon = 0
        render = True
    
    trajectory = generate_episode(env)
    calculate_values(True,trajectory,values)

    
    
# Plotting Every Visit Values
#plt.title("Monte Carlo Every Visit value estimation")
#plt.ylabel("State Values")
#plt.xlabel("Number Of Occurances")
#for (state,action) in list(values.keys())[:10]:
#    plt.plot(range(1, len(values[(state,action)]) + 1), values[(state,action)], label='State - ' + str((state,action)))
#plt.legend(loc='bottom right')
#plt.show()


    