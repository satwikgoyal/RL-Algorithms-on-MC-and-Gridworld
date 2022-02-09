from math import cos, pi
import numpy as np
import random
import matplotlib.pyplot as plt

num_rows = 5
num_cols = 5

v_op = np.array([
    [4.0187, 4.5548, 5.1575, 5.8336, 6.4553],
    [4.3716, 5.0324, 5.8013, 6.6473, 7.3907],
    [3.8672, 4.3900, 0.0000, 7.5769, 8.4637],
    [3.4182, 3.8319, 0.0000, 8.5738, 9.6946],
    [2.9977, 2.9309, 6.0733, 9.6946, 0.0000]
])

num_act = 4

m = 15
d = 2*m + 1

def getAction(s_idx, theta):
  state = getStateRep(s_idx)
  e_score = np.exp(np.sum(theta*state, axis=1))
  if np.sum(e_score) == 0:
    return np.random.randint(num_act)
  softmax_prob = e_score / np.sum(e_score)
  return softmax_prob, random.choices(population=[i for i in range(num_act)], weights=softmax_prob)[0]

def getNextState(s_idx, cur_act):

  def oneDIdx(i, j):
    return int(i*num_rows + j)

  def upIdx(s, i, j):
    if (i == 0) or (s == 22):
      return s
    return oneDIdx(i-1, j)

  def rightIdx(s, i, j):
    if (j == num_cols-1) or (s == 11) or (s == 16):
      return s
    return oneDIdx(i, j+1)
  
  def leftIdx(s, i, j):
    if (j == 0) or (s == 13) or (s == 18):
      return s
    return oneDIdx(i, j-1)

  def downIdx(s, i, j):
    if (i == num_rows-1) or (s == 7):
      return s
    return oneDIdx(i+1, j)
  
  def idIdx(s, i, j):
    return s

  act_dist = {
          0 : {'a': [rightIdx, upIdx, downIdx, idIdx], 'p': [0.8, 0.05, 0.05, 0.1]},
          1 : {'a': [leftIdx, upIdx, downIdx, idIdx], 'p': [0.8, 0.05, 0.05, 0.1]},
          2 : {'a': [upIdx, rightIdx, leftIdx, idIdx], 'p': [0.8, 0.05, 0.05, 0.1]},
          3 : {'a': [downIdx, rightIdx, leftIdx, idIdx], 'p': [0.8, 0.05, 0.05, 0.1]}
      }

  i = s_idx // num_rows
  j = s_idx % num_cols

  return random.choices(population=act_dist[cur_act]['a'], weights=act_dist[cur_act]['p'])[0](s_idx, i, j)

def getReward(s_idx):
  if s_idx == 24:
    return 10
  elif s_idx == 22:
    return -10
  return 0

def getValue(s_idx, w):
  if s_idx == 24:
    return 0
  state = getStateRep(s_idx)
  return np.dot(state, w)

def getStateValue(w):
  v = []
  for s_idx in range(num_rows*num_cols):
    v.append(getValue(s_idx, w))
  return v

def getStateRep(s_idx):
  r = (s_idx // num_rows) / num_rows
  c = (s_idx % num_cols) / num_cols

  return np.array([1] + [cos(i*pi*r) for i in range(1, m+1)] + [
                  cos(i*pi*c) for i in range(1, m+1)])

def actorCritic():

  theta = np.zeros((num_act, d))
  w = np.zeros((d,))
  aw = 0.00005
  at = 0.00005

  gamma = 0.9
  mse_list = []
  act_list = []
  act_count = 0
  while True:

    s_idx = np.random.randint(num_rows*num_cols)
    while s_idx == 12 or s_idx == 17 or s_idx == 24:
        s_idx = np.random.randint(num_rows*num_cols)

    while s_idx != 24:
      
      act_count += 1
      sm_prob, a = getAction(s_idx, theta)
      ns_idx = getNextState(s_idx, a)
      r = getReward(ns_idx)
      delta = r + gamma*getValue(ns_idx, w) - getValue(s_idx, w)
      w += aw * delta * getStateRep(s_idx)
      for i in range(num_act):
        if i == a:
          theta[i] += at * delta * (1-sm_prob[i]) * getStateRep(s_idx)
        else:
          theta[i] += at * delta * -sm_prob[i] * getStateRep(s_idx)

      s_idx = ns_idx

    act_list.append(act_count)

    v = getStateValue(w)
    mse = np.sum((v_op.flatten() - v)**2) / 25
    mse_list.append(mse)
    if mse < 10:
      pi_prob = np.zeros((num_rows*num_cols, 4), dtype="float")
      for s_idx in range(num_rows*num_cols):
        pi_prob[s_idx] = getAction(s_idx, theta)[0]
      break

  return mse_list, act_list, pi_prob

repeat = 20
mse_list = []
act_count = []
min_episodes = float("inf")
pi_sum = np.zeros((num_rows*num_cols, 4), dtype="float")
for i in range(repeat):
    print(i)
    mse, act, cur_pi = actorCritic()
    mse_list.append(mse)
    act_count.append(act)
    if len(mse) < min_episodes:
        min_episodes = len(mse)
    pi_sum += cur_pi

for i in range(repeat):
    mse_list[i] = mse_list[i][:min_episodes]
    act_count[i] = act_count[i][:min_episodes]
mse_list = np.sum(mse_list, axis = 0) / repeat
act_count = np.sum(act_count, axis = 0) / repeat

plt.plot(act_count, range(min_episodes))
plt.xlabel("Number of Actions")
plt.ylabel("Number of Episodes")
plt.savefig("actor_critic_grid_act_vs_ep")
plt.close()

plt.plot(range(min_episodes), mse_list)
plt.xlabel("Number of Episodes")
plt.ylabel("Value Function - Mean Squared Error")
plt.savefig("actor_critic_grid_mse")
plt.close()