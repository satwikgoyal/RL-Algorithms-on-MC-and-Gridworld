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

def true_online_sarsa_mc_tabular(alpha=0.4, gamma=1, λ=0.4, iterations=100, ep=0.4, bins=np.array([20, 5])):
  learning_curve = []
  learning_curve2 = []
  total_steps = 0
  b_x = bins[0]
  b_v = bins[1]
  q = np.zeros((b_x, b_v, 3))
  for i in range(1, iterations+1):
    if ep>0.05:
      ep-=0.01
    car = mountain_car()
    # if i%10==1:
    #   print(f"Done with {i} iterations")
    state = (car.s['x'], car.s['v'])
    x, v = state
    scaled_x = int(((((x - (-1.2))/(0.5 - (-1.2)))*(1-(0))) + (0))//(1/b_x))
    scaled_v = int(((((v - (-0.7))/(0.7 - (-0.7)))*(1-(0))) + (0))//(1/b_v))
    if np.random.rand()<=ep:
      action = int(np.random.rand()//(1/3))-1
    else:
      action = np.argmax(q[scaled_x][scaled_v])-1
    Q_old = 0
    e = np.zeros((b_x, b_v, 3))
    steps = 0
    #episode
    while(True):
      if x >= 0.5:
        # print(f"Reached the goal with {steps} steps!")
        break
      car.move(action)
      next_state = (car.s['x'], car.s['v'])
      x_next, v_next = next_state
      scaled_x_next = int(((((x_next - (-1.2))/(0.5 - (-1.2)))*(1-(0))) + (0))//(1/b_x))
      scaled_v_next = int(((((v_next - (-0.7))/(0.7 - (-0.7)))*(1-(0))) + (0))//(1/b_v))
      next_action = None
      if np.random.rand()<=ep:
        next_action = int(np.random.rand()//(1/3))-1
      else:
        next_action = np.argmax(q[scaled_x_next][scaled_v_next])-1
      reward = car.r()
      delta_Q = q[scaled_x][scaled_v][action+1]-Q_old
      Q_old = q[scaled_x_next][scaled_v_next][next_action+1]
      delta = reward + gamma*q[scaled_x_next][scaled_v_next][next_action+1]-q[scaled_x][scaled_v][action+1]
      e[scaled_x][scaled_v][action+1] = (1-alpha)*e[scaled_x][scaled_v][action+1] + 1
      q += alpha*(delta+delta_Q)*e
      e = gamma*λ*e
      q[scaled_x][scaled_v][action+1] = q[scaled_x][scaled_v][action+1] - alpha*delta_Q
      state = next_state
      action = next_action
      x = x_next
      v = v_next
      scaled_x = scaled_x_next
      scaled_v = scaled_v_next
      total_steps+=1
      steps-=1
    learning_curve.append(total_steps)
    learning_curve2.append(steps)

  return (q, learning_curve, learning_curve2)

q2, lr_curve3, lr_curve4 = true_online_sarsa_mc_tabular(alpha=0.4, gamma=1, λ=0.4, iterations=500, ep=1, bins=np.array([100, 50]))

fig, ax = plt.subplots()
plt.title('Learning Curve 1')
plt.xlabel("Total Steps")
plt.ylabel("Number of Episodes")
ax.scatter(lr_curve3, np.arange(len(lr_curve3)), c='tab:green', alpha=0.05, edgecolors='none')
ax.plot(lr_curve3, np.arange(len(lr_curve3)), c='tab:red')
ax.legend()
ax.grid(True)
plt.show()

fig, ax = plt.subplots()
plt.title('Learning Curve 2')
plt.xlabel("Number of Episodes")
plt.ylabel("Total Return/Episode")
ax.scatter(np.arange(len(lr_curve4)), lr_curve4, c='tab:green', alpha=0.4, edgecolors='none')
# ax.plot(np.arange(len(lr_curve4)), lr_curve4, c='tab:red')
ax.legend()
ax.grid(True)
plt.show()
