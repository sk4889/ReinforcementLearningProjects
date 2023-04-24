# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 23:28:12 2019

@author: Sourabh Kumar
"""

import numpy as np
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2

No_Of_Bandit_Copies = 2000
No_Of_Arms_in_each_Bandit = 10
No_Of_Steps = 1000

# Initializing all the bandit arms initially using an array of size 2000x10 matrix

q_0_1 = np.random.normal(0,1,(No_Of_Bandit_Copies,No_Of_Arms_in_each_Bandit)) # this will return a 2D array of q*a (normal dist of rewards) for all bandit arms
true_optimal_arms = np.argmax(q_0_1,axis=1) # this will return true optimal arms in each bandit

epsilon=[0,0.01,0.1,0.2]

Avg_Reward_each_eps_all_steps = []
optimal_arm_perc_each_eps_all_steps = []

for eps in range(0,len(epsilon)):  
    print('epsilon-',epsilon[eps])
    Q=np.zeros([No_Of_Bandit_Copies,No_Of_Arms_in_each_Bandit]) 
    N=np.zeros([No_Of_Bandit_Copies,No_Of_Arms_in_each_Bandit]) # number of times each arm was pulled # each arm is pulled atleast once, so 1
    Avg_Reward_each_eps_each_step = []
    optimal_arm_perc_each_step_list = []
    
    for step in range(1,No_Of_Steps+1) : # for remaining 1000 steps
        Reward_each_step_list=[] # all rewards in this pull/time-step
        optimal_arm_each_bandit=0 # number of pulls of best arm in this time step
        
        for i in range(No_Of_Bandit_Copies) : # idenotes bandit
            x=np.random.random()    # x is a random value       
            if x < epsilon[eps] : #epsilon = 0.1
                j = np.random.randint(No_Of_Arms_in_each_Bandit) # j denotes arm index               
            else :
                j = np.argmax(Q[i])               
            if j == true_optimal_arms[i] : # To calculate % optimal action
                optimal_arm_each_bandit += 1 # incrementing when opt arm is pulled           
            Reward_each_bandit = np.random.normal(q_0_1[i][j],1)            
            Reward_each_step_list.append(Reward_each_bandit)            
            N[i][j]=N[i][j]+1
            Q[i][j] = (1-1.0/N[i][j])*Q[i][j] + Reward_each_bandit/N[i][j]
            
        Avg_Reward_for_each_step = np.mean(Reward_each_step_list)
        Avg_Reward_each_eps_each_step.append(Avg_Reward_for_each_step) 
        
        optimal_arm_perc_each_step = float(optimal_arm_each_bandit)*100/No_Of_Bandit_Copies
        optimal_arm_perc_each_step_list.append(optimal_arm_perc_each_step)
        
    Avg_Reward_each_eps_all_steps.append(Avg_Reward_each_eps_each_step) 
    optimal_arm_perc_each_eps_all_steps.append(optimal_arm_perc_each_step_list)
   
line_chart1 = plt1.plot(range(1,No_Of_Steps+1),Avg_Reward_each_eps_all_steps[0])
line_chart2 = plt1.plot(range(1,No_Of_Steps+1),Avg_Reward_each_eps_all_steps[1])
line_chart3 = plt1.plot(range(1,No_Of_Steps+1),Avg_Reward_each_eps_all_steps[2])
line_chart4 = plt1.plot(range(1,No_Of_Steps+1),Avg_Reward_each_eps_all_steps[3])
plt1.title('1st Plot - Avg Reward V/s Steps')
plt1.ylabel('Avg. Reward') 
plt1.xlabel('No. of Steps')
plt1.legend(['eps=0','eps=0.01','eps=0.1','eps=0.2'],loc=4)
plt1.show()

line_chart5 = plt2.plot(range(1,No_Of_Steps+1),optimal_arm_perc_each_eps_all_steps[0])
line_chart6 = plt2.plot(range(1,No_Of_Steps+1),optimal_arm_perc_each_eps_all_steps[1])
line_chart7 = plt2.plot(range(1,No_Of_Steps+1),optimal_arm_perc_each_eps_all_steps[2])
line_chart8 = plt2.plot(range(1,No_Of_Steps+1),optimal_arm_perc_each_eps_all_steps[3])
plt2.title('2nd Plot - % Optimal action picked V/s Steps')
plt2.ylabel('% Optimal action picked') 
plt2.xlabel('No. of Steps')
plt2.legend(['eps=0','eps=0.01','eps=0.1','eps=0.2'],loc=4)
plt2.show()
