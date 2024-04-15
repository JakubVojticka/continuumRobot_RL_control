import sys
sys.path.append('../RL')
sys.path.append('../Pytorch')

import torch
import matplotlib.pyplot as plt
from TD3_agent import Agent
from environment import continuumEnv
import math
import time

env = continuumEnv()
agent = Agent(state_dim=4, action_dim=3, seed=10) 

agent.actor_local.load_state_dict(torch.load('../Pytorch/reward_step_error_comparison/model/checkpoint_actor.pth',map_location=torch.device('cpu')))
agent.critic1_local.load_state_dict(torch.load('../Pytorch/reward_step_error_comparison/model/checkpoint_critic.pth',map_location=torch.device('cpu')))
agent.critic2_local.load_state_dict(torch.load('../Pytorch/reward_step_error_comparison/model/checkpoint_critic.pth',map_location=torch.device('cpu')))

state = env.reset()                                                        # Generovanie náhodného počiatočného bodu pre robota a náhodného cieľového bodu
env.start_kappa = [env.kappa1, env.kappa2, env.kappa3]                     # Uloženie počiatočných zakrivení
initial_state = state[0:2]
x_pos = []
y_pos = []

for t in range(750):

    start = time.time()
    action = agent.act(state, add_noise=False)
    
    state, reward, done, _ = env.step_error_comparison(action) # -e^2
    x_pos.append(state[0])
    y_pos.append(state[1])
    print("{}th action".format(t))
    print("Goal Position",state[2:4])

    #print("Error: {0}, Current State: {1}".format(math.sqrt(-1*reward), state)) # for step_2

    if reward < 0:
        print("Error: N/A, Current State:", state)
    else:
        print("Error: {0}, Current State: {1}".format(math.sqrt(reward), state))
    
    print("Action: {0},  Kappas {1}".format(action, [env.kappa1,env.kappa2,env.kappa3]))
    print("Episodic Reward is {}".format(reward))
    print("--------------------------------------------------------------------------------")
    stop = time.time()
    env.time += (stop - start)
    if done:
        break

#==================Vizualizácia==========================#
env.visualization(x_pos, y_pos)
plt.title(f"Initial Position is x: {initial_state[0]} y: {initial_state[1]} & Target Position is x: {state[0]} y: {state[1]}")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.show()

env.close()