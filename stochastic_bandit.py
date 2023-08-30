import random
import math
import numpy as np
import matplotlib.pyplot as plt


class Stochastic_Bandit:
    def __init__(self, mu):
        self.mu = mu
        self.K = len(mu)
    
    def pull(self, id):
        if random.random() < self.mu[id]:
            return 1
        else:   
            return 0


class Epsilon_Greedy:
    def __init__(self, eps):
        self.eps = eps

    def estimate_regrets(self, env, Ts, epoch=50):
        regrets = np.zeros_like(Ts)
        for i, T in enumerate(Ts):
            regret = 0
            for _ in range(epoch):
                regret += self.run(env, int(T))
            regret /= epoch
            regrets[i] = regret
        return regrets
    
    def run(self, env, T):
        K = env.K
        eps = self.eps
        if eps is None:
            c = (env.mu[0] - env.mu[1]) ** 2
            eps = min(1, 2 * K * np.log(T) / (c * T))

        cumsum = 0
        tau = math.floor(T * eps / K)
        rewards = [0] * K
        for i in range(tau):
            for k in range(K):
                r = env.pull(k)
                rewards[k] += r
                cumsum += r

        best_arm = np.argmax(rewards)
        for i in range(T - tau * K):
            cumsum += env.pull(best_arm)
        
        regret = T * max(env.mu) - cumsum
        return regret

    def calc_upper_bound(self, env, Ts):
        upper_bound = np.zeros_like(Ts)
        c = (env.mu[0] - env.mu[1]) ** 2
        for i in range(1, env.K):
            upper_bound += (env.mu[0] - env.mu[i]) * (2 * np.log(Ts) / c + 1)
        return upper_bound
    

class UCB:
    def __init__(self):
        pass

    def estimate_regrets(self, env, Ts, epoch=50):
        regrets = np.zeros_like(Ts)
        for i, T in enumerate(Ts):
            regret = 0
            for _ in range(epoch):
                regret += self.run(env, int(T))
            regret /= epoch
            regrets[i] = regret
        return regrets
    
    def run(self, env, T):
        cumsum = 0
        N = np.zeros(env.K)
        mu_hat = np.zeros(env.K)
        for arm in range(K):
            r = env.pull(arm)
            cumsum += r
            mu_hat[arm] = (mu_hat[arm] * N[arm] + r) / (N[arm] + 1)
            N[arm] += 1

        for t in range(K, T):
            arm = np.argmax(mu_hat + np.sqrt(np.log(t + 1) / 2 / N))
            r = env.pull(arm)
            cumsum += r
            mu_hat[arm] = (mu_hat[arm] * N[arm] + r) / (N[arm] + 1)
            N[arm] += 1
        regret = T * max(env.mu) - cumsum
        return regret

    def calc_upper_bound(self, env, Ts):
        upper_bound = np.zeros_like(Ts)
        for i in range(1, env.K):
            upper_bound += np.log(Ts) / (env.mu[0] - env.mu[i]) / 2
        return upper_bound


if __name__ == '__main__':
    mu = [0.5, 0.4, 0.4, 0.4, 0.4]
    K = len(mu)

    Ts = np.array([1000, 5000, 10000, 20000, 50000, 100000], dtype=np.float64)
    lower_bound = np.zeros_like(Ts)
    for i in range(1, K):
        lower_bound += (mu[0] - mu[i]) * np.log(Ts) / (mu[i] * np.log(mu[i] / mu[0]) + (1 - mu[i]) * np.log((1 - mu[i]) / (1 - mu[0])))

    env = Stochastic_Bandit(mu)
    eps_greedy = Epsilon_Greedy(None)   
    ucb = UCB() 
    
    plt.plot(Ts, eps_greedy.estimate_regrets(env, Ts, 10), color='blue', label='eps-greedy')
    plt.plot(Ts, eps_greedy.calc_upper_bound(env, Ts), linestyle='dashed', color='red', label='eps-greedy (upper bound)')
    plt.plot(Ts, ucb.estimate_regrets(env, Ts, 10), color='orange', label='UCB')
    plt.plot(Ts, ucb.calc_upper_bound(env, Ts), linestyle='dashed', color='purple', label='UCB (upper bound)')
    plt.plot(Ts, lower_bound, linestyle=':', color='green', label='lower bound')

    plt.legend()

    plt.xlabel('T')
    plt.ylabel('regret')
    plt.show()

