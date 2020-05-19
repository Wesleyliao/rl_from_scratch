import numpy as np
import matplotlib.pyplot as plt
import os
import gym

from agents import (
    double_dqn,
    dqn,
    dueling_dqn,
    ac,
    policy_gradient
)


# PERFORMING SINGLE SET
def perform_set(env, agent):

    total_rewards = []

    for i in range(500):
        s = env.reset()
        done = False
        total_reward = 0 

        while not done:
            a = agent.pick_action(s)
            state_prime, reward, done, _ = env.step(a)
            agent.update(s, a, reward, state_prime, done)
            total_reward += reward
            s = state_prime

            # env.render()
        
        if i % 10 == 0: print('--- Iteration',i, 'total reward', total_reward)
        total_rewards.append(total_reward)
    
    return np.array(total_rewards)

# FUNCTION FOR MAKING CHARTS
def make_charts(agent, mean, std):
    n = max(mean.shape)
    index = np.arange(n)
    plt.clf()
    plt.fill_between(index, mean - std, mean + std, alpha=0.2)
    plt.plot(index, mean, '-', linewidth=1, markersize=1, label=None)
    # plt.legend(title=legend_name, loc="best")
    plt.ylabel('Total Reward')
    plt.xlabel('Episode')

    plt.title(agent.description)
    plt.savefig(os.path.join('images', agent.description + '.png'))


# MAIN
def main():

    # SET ENVIRONMENT
    env = gym.make('CartPole-v0')
    # env = gym.make('LunarLander-v2')

    # SET AGENT
    # agent = dqn.DQN(env)
    # agent = double_dqn.DoubleDQN(env)
    # agent = dueling_dqn.DuelingDQN(env)
    agent = policy_gradient.PolicyGradient(env)
    # agent = ac.ActorCritic(env)

    all_run_rewards = []
    for i in range(3):
        print(f'### Starting Run {i+1} ###')
        agent.reset()
        all_run_rewards.append(perform_set(env, agent))
    
    all_run_rewards = np.array(all_run_rewards)
    mean = all_run_rewards.mean(axis=0)
    std = all_run_rewards.std(axis=0)
    make_charts(agent, mean, std)


if __name__ == '__main__':
    main()

