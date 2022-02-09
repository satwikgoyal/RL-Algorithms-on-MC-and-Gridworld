import numpy as np
import matplotlib.pyplot as plt

class mountain_car():
    actions = [-1, 0, 1]

    def __init__(self):
        X0 = np.random.random()*(0.2) -0.6
        self.s = {'x': X0, 'v': 0}

    #next_state
    def move(self, a_t):
        self.s['v'] += 0.001*a_t - 0.0025*np.cos(3*self.s['x'])
        if self.s['v'] > 0.7:
            self.s['v'] = 0.7
        elif self.s['v'] < -0.7:
            self.s['v'] = -0.7

        self.s['x'] += self.s['v']
        if self.s['x']>=0.5:
            self.s['x'] = 0.5
            self.s['v'] = 0
        elif self.s['x']<=-1.2:
            self.s['x'] = -1.2
            self.s['v'] = 0

    def r(self):
        if self.s['x']>=0.5:
            return 0
        return -1

#using a modified cosine fourier basis
def get_features_mc(x, v, d):
    if x>=0.5:
      return np.zeros(d)
    M = (d-1)//2
    state_features = [1]
    # scaled_x = (((x - (-1.2))/(0.5 - (-1.2)))*(1-(0))) + (0)
    # scaled_v = (((v - (-0.7))/(0.7 - (-0.7)))*(1-(0))) + (0)
    scaled_x = (((x - (-1.2))/(0.5 - (-1.2)))*(1-(-1))) + (-1)
    scaled_v = (((v - (-0.7))/(0.7 - (-0.7)))*(1-(-1))) + (-1)

    # for i in range(1, M+1):
    #   state_features.append(np.cos(i*np.pi*scaled_x))
    # for i in range(1, M+1):
    #   state_features.append(np.cos(i*np.pi*scaled_v))
    for i in range(1, M+1):
      state_features.append(np.sin(i*np.pi*scaled_x))
    for i in range(1, M+1):
      state_features.append(np.sin(i*np.pi*scaled_v))
    # for i in range(1, M+1):
    #     state_features.append(x**i)
    # for i in range(1, M+1):
    #     state_features.append(v**i)

    return np.array(state_features)

#get_features_mc(x, v, d)
def sarsa_mc(alpha=0.4, gamma=1, λ=0.2, epsilon=0.2, iterations=100, d = 9):
  w = np.zeros((d, 3))
  learning = []
  learning2 = []
  learning3 = []
  total_steps = 0
  for i in range(iterations):
    # if i%50==0:
    #   print(f"Done with {i} iterations")
    if epsilon>=0.05:
      epsilon -= 0.01
    car = mountain_car()
    state = (car.s['x'], car.s['v'])
    features = get_features_mc(state[0], state[1], d)
    action = None
    if np.random.rand()<=epsilon:
      action = int(np.random.rand()//(1/3))-1
    else:
      action = np.argmax(w.transpose().dot(features)) - 1
    e = np.zeros((d, 3))
    time_steps = 0
    total_reward = 0
    while(time_steps<20000):
      if state[0]>=0.5:
        # print(f"Reached the goal with {time_steps} steps!")
        break
      car.move(action)
      next_state = (car.s['x'], car.s['v'])
      features_next = get_features_mc(next_state[0], next_state[1], d)
      next_action = None
      if np.random.rand()<=epsilon:
        next_action = int(np.random.rand()//(1/3))-1
      else:
        next_action = np.argmax(w.transpose().dot(features_next)) - 1
      reward = car.r()
      delta = reward + gamma*(w[:, next_action+1].dot(features_next)) - w[:, action+1].dot(features)
      e *= gamma*λ
      e[:, action+1] += features
      w += alpha*delta*e
      features = features_next
      action = next_action
      state = next_state
      total_steps += 1
      time_steps += 1
      total_reward += reward

    learning.append(total_steps)
    learning2.append(time_steps)
    learning3.append(total_reward)

  return w, learning, learning2, learning3

w, learning, learning2, learning3 = sarsa_mc(alpha=0.01, gamma=1, λ=0.8, epsilon=1, iterations=1000, d = 9)

fig, ax = plt.subplots()
plt.title('Learning Curve 1')
plt.xlabel("Total Steps")
plt.ylabel("Number of Episodes")
ax.scatter(learning, np.arange(len(learning)), c='tab:purple', alpha=0.05, edgecolors='none')
ax.plot(learning, np.arange(len(learning)), c='tab:blue')
ax.legend()
ax.grid(True)
plt.show()

fig, ax = plt.subplots()
plt.title('Learning Curve 2')
plt.xlabel("Number of Episodes")
plt.ylabel("Total Reward per Episode")
ax.scatter(np.arange(len(learning3)), learning3, c='tab:blue', alpha=0.4, edgecolors='none')
# ax.plot(np.arange(len(learning3)), learning3, c='tab:blue')
ax.legend()
ax.grid(True)
plt.show()
