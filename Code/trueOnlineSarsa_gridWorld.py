import numpy as np
import matplotlib.pyplot as plt

class grid_world():
    def __init__(self):
        self.states = [
                [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
                [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],
                [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)],
                [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4)],
                [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
                ]
        self.states_dict = {
        1: (0, 0),
        2: (0, 1),
        3: (0, 2),
        4: (0, 3),
        5: (0, 4),
        6: (1, 0),
        7: (1, 1),
        8: (1, 2),
        9: (1, 3),
        10: (1, 4),
        11: (2, 0),
        12: (2, 1),
        'Obstacle1': (2, 2),
        13: (2, 3),
        14: (2, 4),
        15: (3, 0),
        16: (3, 1),
        'Obstacle2': (3, 2),
        17: (3, 3),
        18: (3, 4),
        19: (4, 0),
        20: (4, 1),
        21: (4, 2),
        22: (4, 3),
        23: (4, 4)
        }
        self.actions = {"AttemptUp": 0, "AttemptDown": 1, "AttemptLeft": 2, "AttemptRight": 3}
        self.gamma = 0.9
        self.state = self.states[0][0]

    def get_initial_state(self):
        prob = np.random.rand()
        state = min(23, max(1, int(prob//(1/23))))
        return self.states_dict[state]

    def d0(self, state):
        #if not obstacle states
        if state != (2, 2) and state != (3, 2):
            return 1/23
        return 0

    def p(self, state, action, next_state):
        r = state[0]
        c = state[1]
        next_r = next_state[0]
        next_c = next_state[1]

        #coming back to same state surrounding obstacles
        if state==next_state:
            if state==(1, 2):
                if action==self.actions["AttemptDown"]:
                    return 0.9
                elif action!=self.actions["AttemptUp"]:
                    return 0.15
            if state==(4, 2):
                if action==self.actions["AttemptRight"] or action==self.actions["AttemptLeft"]:
                    return 0.2
                elif action==self.actions["AttemptUp"]:
                    return 0.9
            elif (state==(2, 3) or state==(3, 3)):
                if action==self.actions["AttemptLeft"]:
                    return 0.9
                elif action!=self.actions["AttemptRight"]:
                    return 0.15
            elif (state==(3, 1) or state==(2, 1)):
                if action==self.actions["AttemptRight"]:
                    return 0.9
                elif action!=self.actions["AttemptLeft"]:
                    return 0.15

        #terminating state
        if state == (4,4):
            return 0

        #obstacles
        if state == (3, 2) or state == (2, 2):
            return 0

        if next_c == 2 and (next_r == 2 or next_r == 3):
            return 0

        if action == self.actions["AttemptUp"]:
            if next_r == r == 0:
                if (c == next_c == 0) or (c == next_c == 4):
                    return 0.95
                if next_c == c-1:
                    return 0.05
                elif next_c == c+1:
                    return 0.05
                elif next_c == c:
                    return 0.90
            if (c == next_c == 0) or (c == next_c == 4):
                if next_r == r:
                    return 0.15
            if r == next_r:
                if next_c == c-1:
                    return 0.05
                elif next_c == c+1:
                    return 0.05
                elif next_c == c:
                    return 0.10
            if next_r == r-1:
                if next_c == c:
                    return 0.80
        elif action == self.actions["AttemptDown"]:
            if (next_r == r == 4):
                if (c == next_c == 0) or (c == next_c == 4):
                    return 0.95
                if next_c == c-1:
                    return 0.05
                elif next_c == c+1:
                    return 0.05
                elif next_c == c:
                    return 0.90
            if (c == next_c == 0) or (c == next_c == 4):
                if next_r == r:
                    return 0.15
            if r == next_r:
                if next_c == c-1:
                    return 0.05
                elif next_c == c+1:
                    return 0.05
                elif next_c == c:
                    return 0.10
            if next_r == r+1:
                if next_c == c:
                    return 0.8
        elif action == self.actions["AttemptRight"]:
            if (next_c == c == 4):
                if (r == next_r == 0) or (r == next_r == 4):
                    return 0.95
                if next_r == r-1:
                    return 0.05
                elif next_r == r+1:
                    return 0.05
                elif next_r == r:
                    return 0.90
            if (r == next_r == 0) or (r == next_r == 4):
                if next_c == c:
                    return 0.15
            if c == next_c:
                if next_r == r-1:
                    return 0.05
                elif next_r == r+1:
                    return 0.05
                elif next_r == r:
                    return 0.10
            if next_c == c+1:
                if next_r == r:
                    return 0.8
        elif action == self.actions["AttemptLeft"]:
            if (next_c == c == 0):
                if (r == next_r == 0) or (r == next_r == 4):
                    return 0.95
                if next_r == r-1:
                    return 0.05
                elif next_r == r+1:
                    return 0.05
                elif next_r == r:
                    return 0.90
            if (r == next_r == 0) or (r == next_r == 4):
                if next_c == c:
                    return 0.15
            if c == next_c:
                if next_r == r-1:
                    return 0.05
                elif next_r == r+1:
                    return 0.05
                elif next_r == r:
                    return 0.10
            if next_c == c-1:
                if next_r == r:
                    return 0.8
        return 0

    def get_next_state(self, state, action):
        r = state[0]
        c = state[1]
        next_state = None
        prob = np.random.random()
        if action == self.actions["AttemptUp"]:
            if prob<=0.05:
                next_state = (r, max(0, c-1))
            elif prob<=0.85:
                next_state = (max(r-1, 0), c)
            elif prob<=0.9:
                next_state = (r, min(c+1, 4))
            else:
                next_state = state
        elif action == self.actions["AttemptDown"]:
            if prob<=0.05:
                next_state = (r, min(c+1, 4))
            elif prob<=0.85:
                next_state = (min(r+1, 4), c)
            elif prob<=0.9:
                next_state = (r, max(c-1, 0))
            else:
                next_state = state
        elif action == self.actions["AttemptRight"]:
            if prob<=0.05:
                next_state = (max(r-1, 0), c)
            elif prob<=0.85:
                next_state = (r, min(c+1, 4))
            elif prob<=0.9:
                next_state = (min(r+1, 4), c)
            else:
                next_state = state
        else: #action == AttemptLeft
            if prob<=0.05:
                next_state = (min(r+1, 4), c)
            elif prob<=0.85:
                next_state = (r, max(0, c-1))
            elif prob<=0.9:
                next_state = (max(0, r-1), c)
            else:
                next_state = state

        if next_state == (2, 2) or next_state == (3, 2):
            return state

        return next_state

    def R(self, next_state):
        if next_state == (4, 4):
            return 10
        elif next_state == (4, 2):
            return -10

        return 0

    def set_gamma(self, new_gamma):
        self.gamma = new_gamma

    def get_gamma(self):
        return self.gamma

    def get_actions(self):
        return self.actions

def true_online_sarsa_gw(alpha=0.4, gamma=0.9, λ=0.4, iterations=100, ep=0.4, q=np.zeros((5, 5, 4))):
  q[4][4] = [0, 0, 0, 0]
  q[2][2] = [0, 0, 0, 0]
  q[3][2] = [0, 0, 0, 0]
  v_opt = np.array([
    [4.0187, 4.5548, 5.1575, 5.8336, 6.4553],
    [4.3716, 5.0324, 5.8013, 6.6473, 7.3907],
    [3.8672, 4.3900, 0.0000, 7.5769, 8.4637],
    [3.4182, 3.8319, 0.0000, 8.5738, 9.6946],
    [2.9977, 2.9309, 6.0733, 9.6946, 0.0000]
    ])
  world = grid_world()
  learning_curve = []
  learning_curve2 = []
  learning_curve3 = []
  total_steps = 0
  for i in range(1, iterations+1):
    # if i%10==1:
    #   print(f"Done with {i} iterations")
    # if ep>0.05:
    #   ep-=0.05
    # state = world.get_initial_state()
    state = (0, 0)
    r, c = state
    if np.random.rand()<=ep:
      action = int(np.random.rand()//0.25)
    else:
      action = np.argmax(q[r][c])
    e = np.zeros((5, 5, 4))
    Q_old = 0
    #episode
    power = 0
    total_reward = 0
    while(state != (4, 4)):
      next_state = world.get_next_state(state, action)
      r_next, c_next = next_state
      next_action = -1
      if np.random.rand()<=ep:
        next_action = int(np.random.rand()//0.25)
      else:
        next_action = np.argmax(q[r_next][c_next])
      reward = world.R(next_state)
      delta_Q = q[r][c][action]-Q_old
      Q_old = q[r_next][c_next][next_action]
      delta = reward + gamma*q[r_next][c_next][next_action]-q[r][c][action]
      e[r][c][action] = (1-alpha)*e[r][c][action] + 1
      q += alpha*(delta+delta_Q)*e
      e = gamma*λ*e
      q[r][c][action] = q[r][c][action] - alpha*delta_Q
      state = next_state
      action = next_action
      r = r_next
      c = c_next
      total_steps+=1
      total_reward+= (gamma**power)*reward
      power+=1

    v = []
    for r in range(5):
      v.append([])
      for c in range(5):
        v[r].append(np.max(q[r][c]))
    learning_curve.append(total_steps)
    learning_curve2.append(total_reward)
    learning_curve3.append((np.sum(np.square(v_opt-v)))/25)
  return (q, learning_curve, learning_curve2, learning_curve3)

q, lr_curve2, lr_curve1, lr_curve3 = true_online_sarsa_gw(alpha=0.4, gamma=0.9, λ=0.4, iterations=1000, ep=0.05, q=np.ones((5, 5, 4))*1000)
v = []
policy = []
for r in range(5):
  v.append([])
  policy.append([])
  for c in range(5):
    v[r].append(np.max(q[r][c]))
    policy[r].append(np.argmax(q[r][c]))
print(np.array(v))
print(np.array(policy))

v_opt = np.array([
    [4.0187, 4.5548, 5.1575, 5.8336, 6.4553],
    [4.3716, 5.0324, 5.8013, 6.6473, 7.3907],
    [3.8672, 4.3900, 0.0000, 7.5769, 8.4637],
    [3.4182, 3.8319, 0.0000, 8.5738, 9.6946],
    [2.9977, 2.9309, 6.0733, 9.6946, 0.0000]
    ])
print(f"MSE: {np.sum(np.square(v_opt-v))/25}")

fig, ax = plt.subplots()
plt.title('Learning Curve 1')
plt.xlabel("Total Steps")
plt.ylabel("Number of Episodes")
ax.scatter(lr_curve2, np.arange(len(lr_curve2)), c='tab:green', alpha=0.05, edgecolors='none')
ax.plot(lr_curve2, np.arange(len(lr_curve2)), c='tab:red')
ax.legend()
ax.grid(True)
plt.show()

fig, ax = plt.subplots()
plt.title('Learning Curve 2')
plt.xlabel("Number of Episodes")
plt.ylabel("Total Return/Episode")
ax.scatter(np.arange(len(lr_curve1)), lr_curve1, c='tab:green', alpha=0.4, edgecolors='none')
# ax.plot(np.arange(len(lr_curve1)), lr_curve1, c='tab:red')
ax.legend()
ax.grid(True)
plt.show()

fig, ax = plt.subplots()
plt.title('Learning Curve 3')
plt.xlabel("Number of Episodes")
plt.ylabel("Mean Squared Error")
ax.scatter(np.arange(len(lr_curve3)), lr_curve3, c='tab:green', alpha=0.05, edgecolors='none')
ax.plot(np.arange(len(lr_curve3)), lr_curve3, c='tab:red')
ax.legend()
ax.grid(True)
plt.show()
