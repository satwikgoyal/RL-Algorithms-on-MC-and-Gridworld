from math import cos, pi
import numpy as np
import random
import matplotlib.pyplot as plt

x_min = -1.2
x_max = 0.5
v_min = -0.7
v_max = 0.7

num_act = 3

m = 20
d = 2*m + 1

def getAction(xt, vt, theta):
  state = getStateRep(xt, vt)
  e_score = np.exp(np.sum(theta*state, axis=1))
  if np.sum(e_score) == 0:
    return np.random.randint(num_act)
  softmax_prob = e_score / np.sum(e_score)
  return softmax_prob, random.choices(population=[0, 1, 2], weights=softmax_prob)[0]

def getNextState(xt, vt, a):
    next_vt = vt + 0.001*a - 0.0025*cos(3*xt)
    next_xt = xt + next_vt
    next_vt = np.clip(next_vt, v_min, v_max)
    next_xt = np.clip(next_xt, x_min, x_max)
    if next_xt == x_min or next_xt == x_max:
      next_vt = 0

    return next_xt, next_vt

def getReward(xt):
    if xt == 0.5:
      return 0
    return -1

def getValue(xt, vt, w):
  if xt == 0.5:
    return 0
  state = getStateRep(xt, vt)
  return np.dot(state, w)

def getStateRep(xt, vt):
  xt = (xt - x_min) / (x_max - x_min)
  vt = (vt - v_min) / (v_max - v_min)

  return np.array([1] + [cos(i*pi*xt) for i in range(1, m+1)] + [
                  cos(i*pi*vt) for i in range(1, m+1)])
  
def getAvgReturn(theta):

    j = 0
    num_ep = 10
    for ep in range(num_ep):
        disc_rew = 0
        xt = np.random.uniform(-0.6, -0.4)
        vt = 0
        t = 0
        while t < 1000:
            _, action = getAction(xt, vt, theta)
            xt, vt = getNextState(xt, vt, action-1)
            if xt == 0.5:
                break
            else:
                disc_rew += -1
            t += 1
        j += disc_rew
    return j / num_ep

def actorCritic(num_trials):

  theta = np.zeros((num_act, d))
  w = np.zeros((d,))

  aw = 0.0001
  at = 0.0001

  j = []
  cur_act_list = []
  cur_act_count = 0
  for it in range(num_trials):

    xt = random.uniform(x_min, x_max)
    while xt == 0.5:
      xt = random.uniform(x_min, x_max)
    vt = random.uniform(v_min, v_max)

    t = 0
    while t < 1000:
      
      cur_act_count += 1
      sm_prob, a = getAction(xt, vt, theta)
      next_xt, next_vt = getNextState(xt, vt, a-1)
      r = getReward(xt)
      delta = r + getValue(next_xt, next_vt, w) - getValue(xt, vt, w)
      w += aw * delta * getStateRep(xt, vt)
      for i in range(num_act):
        if i == a:
          theta[i] += at * delta * (1-sm_prob[i]) * getStateRep(xt, vt)
        else:
          theta[i] += at * delta * -sm_prob[i] * getStateRep(xt, vt)

      if next_xt == 0.5:
        break
      xt = next_xt
      vt = next_vt
      t += 1 

    cur_act_list.append(cur_act_count)
    if it % 50 == 0:
      j.append(getAvgReturn(theta))

  return j, cur_act_list

repeat = 20
num_trials = 2000
j_list = []
act_count = []
for i in range(repeat):
    print(i)
    j, act_list = actorCritic(num_trials)
    print(j)
    j_list.append(j)
    act_count.append(act_list)

j_avg = np.mean(j_list, axis = 0)
j_std = np.std(j_list, axis = 0)

act_count = np.sum(act_count, axis = 0) / repeat

plt.plot(act_count, range(num_trials))
plt.xlabel("Number of Actions")
plt.ylabel("Number of Episodes")
plt.savefig("actor_critic_mc_act_vs_ep.png")
plt.close()

plt.errorbar(range(j_avg.shape[0]), j_avg, j_std)
plt.xlabel('Number of runs')
plt.ylabel('Mean Avg Return')
plt.savefig('actor_critic_mc_lc.png')