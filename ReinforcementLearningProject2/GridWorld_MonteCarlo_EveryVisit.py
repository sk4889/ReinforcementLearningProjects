#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 01:48:46 2019

@author: sourabhkumar
"""

# -*- coding: utf-8 -*-


import numpy as np

import random

import matplotlib.pyplot as plt

gamma = 0.95
gridSize = 4

terminationStates = [[0,0], [gridSize-1, gridSize-1]]

actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]

numIterations = 50000
# initialization
V = np.zeros((gridSize, gridSize))
returns = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
deltas = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
states_values = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]
a=numIterations/14

action_prob = {(0,1):  [0.1,0.1,0.1,0.7], (0,2) : [0.1,0.1,0.1,0.7], (0,3) : [0.1,0.4,0.1,0.4],
               (1,0) : [0.7,0.1,0.1,0.1], (1,1) : [0.4,0.1,0.1,0.4], (1,2) : [0.1,0.4,0.1,0.4],
               (1,3) : [0.1,0.7,0.1,0.1], (2,0) : [0.7,0.1,0.1,0.1], (2,1) : [0.4,0.1,0.4,0.1],
               (2,2) : [0.1,0.4,0.4,0.1], (2,3) : [0.1,0.7,0.1,0.1], (3,0) : [0.4,0.1,0.4,0.1],
               (3,1) : [0.1,0.1,0.7,0.1], (3,2) : [0.1,0.1,0.7,0.1]}


def EpisodeGenerator(initState,prob):
    episode = []
    while True:
        if initState in terminationStates:
            return episode
        prob=action_prob[initState[0],initState[1]]
        action = random.choices(actions,prob) 
        finalState = list(np.array(initState)+np.array(action[0]))
        if -1 in finalState or gridSize in finalState:
            finalState = initState
        if finalState in ([0, 0],[3, 3]):
            rewardSize = 0
        else:
            rewardSize = -1 
        episode.append([initState, action, rewardSize, finalState]) 
        initState = finalState

for iter in range(numIterations): 
    if 0 <= iter < a:
        initState = [0, 1]
        prob=action_prob[initState[0],initState[1]]
    elif a <= iter < 2*a:
        initState = [0,2]
        prob=action_prob[initState[0],initState[1]]
    elif 2*a <= iter < 3*a:
        initState = [0,3]
        prob=action_prob[initState[0],initState[1]]
    elif 3*a <= iter < 4*a:        
        initState = [1,0]
        prob=action_prob[initState[0],initState[1]]
    elif 4*a <= iter < 5*a:
        initState = [1,1]
        prob=action_prob[initState[0],initState[1]]
    elif 5*a <= iter < 6*a:
        initState = [1,2]
        prob=action_prob[initState[0],initState[1]]
    elif 6*a <= iter < 7*a:
        initState = [1,3]
        prob=action_prob[initState[0],initState[1]]
    elif 7*a <= iter < 8*a:
        initState = [2,0]
        prob=action_prob[initState[0],initState[1]]
    elif 8*a <= iter < 9*a:
        initState = [2,1]
        prob=action_prob[initState[0],initState[1]]
    elif 9*a <= iter < 10*a:
        initState = [2,2]
        prob=action_prob[initState[0],initState[1]]
    elif 10*a <= iter < 11*a:
        initState = [2,3]
        prob=action_prob[initState[0],initState[1]]
    elif 11*a <= iter < 12*a:
        initState = [3,0]
        prob=action_prob[initState[0],initState[1]]
    elif 12*a <= iter < 13*a:
        initState = [3,1]
        prob=action_prob[initState[0],initState[1]]
    elif 13*a <= iter < 14*a:
        initState = [3,2]
        prob=action_prob[initState[0],initState[1]]

    #Designing episode by calling generate episode method    
    episode = EpisodeGenerator(initState,prob) 
    #Working on returns and Values (G & V)
    G = 0
    for i, step in enumerate(episode[::-1]): #reverses the episode list for loop
        G = gamma*G + step[2]
        index = (step[0][0], step[0][1])
        returns[index].append(G)
        newValue = np.average(returns[index])
        deltas[index[0], index[1]].append(np.abs(V[index[0], index[1]]-newValue))
        V[index[0], index[1]] = newValue
        states_values[index[0], index[1]].append(np.abs(V[index[0], index[1]]))
#Printing final values for all the states 
for i in range(gridSize):
    for j in range(gridSize):
        if (i == 0 and j == 0) or (i == 3 and j == 3):
            print('Final Value for target state in grid [',i,'][',j,']-',V[i][j])
        else:    
            print('Final Value for intermediate state in grid[',i,'][',j,']-',V[i][j])
        
plt.figure(figsize=(20,10))
plt.title('Plot - Avg Retruns V/s Iterations')
plt.ylabel('Average Retruns') 
plt.xlabel('No. of Iterations')
all_series = [list(x)[:50000] for x in states_values.values()]
for series in all_series:
    plt.plot(series)
plt.legend(['1','2','3','4','5','6','7','8','9','10','11','12','13','14'],loc=1)
plt.show()
