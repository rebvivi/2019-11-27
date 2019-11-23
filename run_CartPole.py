import gym
import cartpole_gym
from RL_brain import DeepQNetwork
import matplotlib.pyplot as plt
import math
env = gym.make('CartPoleEnv11-v0') #定義使用 gym 的哪一個環境
env = env.unwrapped #不做這個會有很多限制

print(env.action_space)  #查看這個環境中可用的 action 有多少個
print(env.observation_space)   #查看這個環境中可用的 state 的 observation 有多少個
print(env.observation_space.high) #查看 observation 的最大值
print(env.observation_space.low) #查看 observation 的最小值
 
RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)
def get_epsilon(t):
    ''' decrease the exploration rate at each episode '''
    return max(min_epsilon, min(1, 1.0 - math.log10((t + 1) / ada_divisor)))
min_epsilon = 0.1           # exploration rate
ada_divisor = 25     
total_steps = 0 #紀錄 step 數
rewards=[]
n_episodes=300
#跑 100 個 episode , 每一個 episode 都是一次任務嘗試
for i_episode in range(n_episodes):
    #取得回合 i_episode 的第一個 observation
    observation = env.reset()  #讓 environment 回到初始狀態
    ep_r = 0 #累計各個 episode 的 reward
    epsilon=get_epsilon(i_episode)
    while True:
        env.render() #刷新 environment , 呈現 environment ,

        action = RL.choose_action(observation,epsilon) #根據現在狀態來採取一個行為

       #take action
        observation_, reward, done, info = env.step(action) #根據採取的行為得到了回饋

        # the smaller theta and closer to center the better
        x, x_dot, theta, theta_dot = observation_
        # x 是車子的水平位移 ,  車月偏離中心 , r1 越小 

        RL.store_transition(observation, action, reward, observation_) #儲存 之前state , action , reward ,下一個 state

        ep_r += reward #累計 reward
        if total_steps > 1000:
            RL.learn() #進行學習

        if done: #回合結束就跳到下一個回合
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1
    rewards.append(ep_r)


# PLOT RESULTS
x = range(n_episodes)
plt.plot(x, rewards)
plt.xlabel('episode')
plt.ylabel('Training cumulative reward')
plt.savefig('Q_learning_CART.png', dpi=300)
plt.show()
RL.plot_cost()