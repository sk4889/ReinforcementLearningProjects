# ReinforcementLearningProjects
Lunar Lander V2 - 
Goal: Land on the landing pad with zero speed and least fuel consumption. Achieve 200 points
Rewards:
•	Winning the game is 200 points
•	Reward for moving from the top of the screen to landing pad and zero speed is about 100 to 140 points.
•	If lander moves away from landing pad it loses reward back. 
•	Episode finishes if the lander crashes -100 points or comes to rest +100 points
•	Each leg ground contact is +10
•	Firing main engine is -0.3 points each frame
Action Space: 
Four discrete actions available, 
1.	Do nothing 
2.	Fire left orientation engine
3.	Fire main engine
4.	fire right orientation engine

Note: Landing outside landing pad is possible. Fuel is infinite

Sample run below from the Open AI Webpage
