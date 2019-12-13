""" numpy implementation of Moody 2001 DRL trader by Matt Pearce """
import numpy as np


def normalise(x):
    return (x - x.mean()) / x.std()


class Env():

    def __init__(self, r, mu, TC, T, m, z):
        self.quantity = mu
        self.transaction_costs = TC
        self.F_prev = 0.0
        self.T = T
        self.m = m
        self.r = r # returns  z{t} - z{t-1}
        self.z = z

    def get_reward(self, r_t, F_t):
        r_t = self.z[self.t+self.m-1] - self.z[self.t+self.m-1-1]
        return self.quantity * (self.F_prev * r_t - self.transaction_costs * abs(F_t - self.F_prev))

    def get_observation(self):
        return normalise(self.r)[self.t : self.t + self.m]

    def step(self, action):
        # print(self.t, 'F =', action)
        self.t += 1

        reward = self.get_reward(self.r[self.t+self.m-2], action)
        self.F_prev = action

        observation = self.get_observation()

        done = self.t >= self.T

        return observation, reward, done

    def reset(self):
        self.t = 1
        self.F_prev = 0.0
        return self.get_observation()




class MoodyDRLAgent():

    def __init__(self, r, m, TC, mu):
        # self.theta = np.ones(m+2) # params are [b,theta(m),u]
        self.theta = (np.random.rand(m+2)*2-1.0)*np.sqrt(6./(m+2))
        self.rho = 0.04      # learning rate
        self.m = m
        self.reset()
        self.TC = TC
        self.mu = mu
        self.r = r
        self.reset()

    def reset(self):
        self.F = [0.]
        self.I = [0.]

    def get_features(self, observation, F_prev):
        # obs_norm = normalise(observation)
        return np.concatenate([[1], observation, [F_prev]])

    def get_action(self, observation):
        It = self.get_features(observation, self.F[-1])
        self.I.append(It)
    #     print ('get_action', b, u, Ft_prev, I)
        Ft = np.tanh(np.dot(self.theta, It))
        self.F.append(Ft)
        return Ft

    def fit(self):
        T = len(self.F)
        dF = np.zeros([self.m + 2, T])

        for i in range(1, T):
            It = self.I[i]
            sech2 = 1 - np.power(np.tanh(np.dot(self.theta, It)), 2)
            dF[:,i] = sech2 * (It + self.theta[-1] * dF[:,i-1])

        F = np.array(self.F)

        dRtdFt = -self.mu * self.TC * np.sign(F[1:] - F[0:-1])
        dRtdFt1 = self.mu * (self.r[self.m:self.m+T-1] + self.TC * np.sign(F[1:] - F[0:-1]))

        dUt = dRtdFt * dF[:,1:] + dRtdFt1 * dF[:,0:-1]

        self.theta = self.theta + self.rho * np.sum(dUt,1)






#
# training
#
def training(tick_data):
    T = 1000        # training size
    m = 50          # number of prices in input feature window
    mu = 1.         # trade quantity
    TC = 0.002      # transaction costs


    z = tick_data[:T + m] # training prices
    r = (z[1:] - z[:-1]) #/ z[1:]

    env = Env(r=r, T=T, TC=TC, mu=mu, m=m, z=z)

    agent = MoodyDRLAgent(r, m, TC=TC, mu=mu)


    num_epochs = 100
    epoch = 1
    total_reward = []
    R = []
    observation = env.reset()
    while (True):

        action = agent.get_action(observation)

        observation, reward, done = env.step(action)
        R.append(reward)

        if done:
            # print stats
            print (epoch, 'total reward=', np.sum(R))
            total_reward.append(np.sum(R))

            # train policy (gradient ascent)
            agent.fit()

            # start next epoch/episode
            epoch += 1
            if epoch > num_epochs:
                break
            observation = env.reset()
            R = []
            agent.reset()

    # output some stats
    print (agent.theta)
    print ("Buy and Hold PnL", z[-1]-z[0], 'v', total_reward[-1])
    import matplotlib.pyplot as plt
    ax=plt.subplot(3, 1, 1)
    plt.plot(total_reward)
    ax.set_title("Cum Reward per epoch")
    ax=plt.subplot(3, 1, 2)
    plt.plot(z[m:])
    ax.set_title("Time-series (prices)")
    ax=plt.subplot(3, 1, 3)
    plt.plot(np.round(agent.F))
    ax.set_title("Trading signals")
    plt.show()

    return agent, env


if __name__ == '__main__':
    import moody_ts_gen
    np.random.seed(0)
    tick_data = moody_ts_gen.generate_timeseries(10000)
    training(tick_data)

