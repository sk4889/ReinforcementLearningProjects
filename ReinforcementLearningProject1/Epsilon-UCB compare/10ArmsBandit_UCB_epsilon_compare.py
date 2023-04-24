# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 13:02:40 2019

@author: Sourabh Kumar

UCB-Epsilon compare for C=2 and e=0.1
"""

import numpy as np
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2

No_Of_Bandit_Copies = 2000
No_Of_Arms_in_each_Bandit = 10
No_Of_Steps = 1000

# Initializing all the bandit arms initially using an array of size 2000x10 matrix using normal disdtribution
q_0_1 = np.random.normal(0,1,(No_Of_Bandit_Copies,No_Of_Arms_in_each_Bandit)) 
Qi=np.random.normal(q_0_1,1) # initial pulling of all arms
avg_Qi=np.mean(Qi)
# this will return true optimal arms in each bandit
true_optimal_arms = np.argmax(q_0_1,axis=1) 

# controls degree of exploration
ucb_list=[1,2,3]


Avg_Reward_each_ucb_all_steps = []
#Avg_Reward_each_ucb_all_steps.append(0)
#Avg_Reward_each_ucb_all_steps.append(avg_Qi)

optimal_arm_perc_each_ucb_all_steps = []
# comparing with epsilon-greedy method 
eps=0.1 

Avg_Reward_each_eps_all_steps = []
optimal_arm_perc_each_eps_all_steps = []

for ucb in range(0,len(ucb_list)) : # for ucb_list
    print ('Current value of ucb : ', ucb_list[ucb])
    # Estimates of rewards initiallized to 0 for all the arms
    Q_ucb=np.zeros([No_Of_Bandit_Copies,No_Of_Arms_in_each_Bandit]) 
    # number of times each arm is pulled, initialized to 0 for all the arms
    N_ucb=np.zeros([No_Of_Bandit_Copies,No_Of_Arms_in_each_Bandit]) 
    # Estimates of rewards initiallized to 0 for all the arms    
    Q_eps=np.zeros([No_Of_Bandit_Copies,No_Of_Arms_in_each_Bandit])
    # number of times each arm is pulled, initialized to 0 for all the arms
    N_eps=np.zeros([No_Of_Bandit_Copies,No_Of_Arms_in_each_Bandit])
    
    
    Avg_Reward_each_ucb_each_step = []
    optimal_arm_perc_each_step_list_ucb = []
    
    Avg_Reward_each_eps_each_step = []
    optimal_arm_perc_each_step_list_eps = []
    
    for step in range(1,No_Of_Steps+1) : # for 1000 steps, step denotes each step
        Reward_each_step_list_ucb=[] # all rewards in this pull/time-step for ucb
        optimal_arm_each_bandit_ucb=0 # number of pulls of best arm in this time step for ucb
        
        Reward_each_step_list_eps=[] # all rewards in this pull/time-step for eps
        optimal_arm_each_bandit_eps=0 # number of pulls of best arm in this time step for eps
        
        for bandit in range(No_Of_Bandit_Copies) : # for all bandits
            #ucb reward and estimate calculation
            #print(Q_ucb[bandit])
            optimal_pulled_arm_idx_ucb=np.argmax(Q_ucb[bandit]+ucb_list[ucb]*np.sqrt(np.log(step)/(N_ucb[bandit]+1))) # arm index of maximum estimate amongst all the arms
            #print(optimal_pulled_arm_idx_ucb)
            if optimal_pulled_arm_idx_ucb==true_optimal_arms[bandit]:
                optimal_arm_each_bandit_ucb = optimal_arm_each_bandit_ucb+1   
            
            Reward_each_bandit_ucb = q_0_1[bandit][optimal_pulled_arm_idx_ucb]+ np.random.normal(0,1)                 
            N_ucb[bandit][optimal_pulled_arm_idx_ucb] = N_ucb[bandit][optimal_pulled_arm_idx_ucb] + 1
            Q_ucb[bandit][optimal_pulled_arm_idx_ucb] = (1-1.0/N_ucb[bandit][optimal_pulled_arm_idx_ucb])*Q_ucb[bandit][optimal_pulled_arm_idx_ucb] + Reward_each_bandit_ucb/N_ucb[bandit][optimal_pulled_arm_idx_ucb]               
            Reward_each_step_list_ucb.append(Reward_each_bandit_ucb) 
            #print(Q_ucb[bandit])
            
            #epsilon greedy reward and estimate calculation when ucb_list[ucb]=2
            if ucb==1: 
                x=np.random.random()    # x is a random value       
                if x < eps : #epsilon = 0.1
                    arm_idx = np.random.randint(No_Of_Arms_in_each_Bandit) # j_eps denotes arm index               
                else :
                    arm_idx = np.argmax(Q_eps[bandit])               
                if arm_idx == true_optimal_arms[bandit] : # To calculate % optimal action
                    optimal_arm_each_bandit_eps += 1 # incrementing when opt arm is pulled           
                Reward_each_bandit_eps = np.random.normal(q_0_1[bandit][arm_idx],1)            
                Reward_each_step_list_eps.append(Reward_each_bandit_eps)            
                N_eps[bandit][arm_idx]=N_eps[bandit][arm_idx]+1
                Q_eps[bandit][arm_idx] = (1-1.0/N_eps[bandit][arm_idx])*Q_eps[bandit][arm_idx] + Reward_each_bandit_eps/N_eps[bandit][arm_idx]
                
        Avg_Reward_for_each_step_ucb = np.mean(Reward_each_step_list_ucb)
        Avg_Reward_each_ucb_each_step.append(Avg_Reward_for_each_step_ucb)         
            
        optimal_arm_perc_each_step_ucb = float(optimal_arm_each_bandit_ucb)*100/No_Of_Bandit_Copies
        optimal_arm_perc_each_step_list_ucb.append(optimal_arm_perc_each_step_ucb)  
        
        if ucb==1:
            Avg_Reward_for_each_step_eps = np.mean(Reward_each_step_list_eps)
            Avg_Reward_each_eps_each_step.append(Avg_Reward_for_each_step_eps)         
            
            optimal_arm_perc_each_step_eps = float(optimal_arm_each_bandit_eps)*100/No_Of_Bandit_Copies
            optimal_arm_perc_each_step_list_eps.append(optimal_arm_perc_each_step_eps)
    
    Avg_Reward_each_ucb_all_steps.append(Avg_Reward_each_ucb_each_step)

    optimal_arm_perc_each_ucb_all_steps.append(optimal_arm_perc_each_step_list_ucb)
    
    Avg_Reward_each_eps_all_steps.append(Avg_Reward_each_eps_each_step) 

    optimal_arm_perc_each_eps_all_steps.append(optimal_arm_perc_each_step_list_eps)


#print(Avg_Reward_each_ucb_all_steps[1])
line_chart1 = plt1.plot(range(1,No_Of_Steps+1),Avg_Reward_each_ucb_all_steps[0])
line_chart2 = plt1.plot(range(1,No_Of_Steps+1),Avg_Reward_each_ucb_all_steps[1])
line_chart3 = plt1.plot(range(1,No_Of_Steps+1),Avg_Reward_each_eps_all_steps[1])
line_chart4 = plt1.plot(range(1,No_Of_Steps+1),Avg_Reward_each_ucb_all_steps[2])
plt1.title('1st Plot - Avg Reward UCB and eps greedy')
plt1.ylabel('Avg. Reward') 
plt1.xlabel('No. of Steps')
plt1.legend(['ucb=1','ucb=2','eps=0.1','ucb=3'],loc=4)
plt1.show()
#
line_chart5 = plt2.plot(range(1,No_Of_Steps+1),optimal_arm_perc_each_ucb_all_steps[0])
line_chart6 = plt2.plot(range(1,No_Of_Steps+1),optimal_arm_perc_each_ucb_all_steps[1])
line_chart7 = plt2.plot(range(1,No_Of_Steps+1),optimal_arm_perc_each_eps_all_steps[1])
line_chart8 = plt2.plot(range(1,No_Of_Steps+1),optimal_arm_perc_each_ucb_all_steps[2])
plt2.title('2nd Plot - % Optimal actions ECB and eps greedy')
plt2.ylabel('% Optimal action picked') 
plt2.xlabel('No. of Steps')
plt2.legend(['ucb=1','ucb=2','eps=0.1','ucb=3'],loc=4)
plt2.show()
