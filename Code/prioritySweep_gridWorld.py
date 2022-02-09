import heapq
import random
import numpy as np
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

def getState(cur_state, cur_act):

    act_to_func = {
            0 : rightIdx,
            1 : leftIdx,
            2 : upIdx,
            3 : downIdx
        }

    i = cur_state // num_rows
    j = cur_state % num_rows

    return act_to_func[cur_act](cur_state, i, j)

def getAllStates(cur_state, cur_act):
    st_list = []
    i = cur_state // num_rows
    j = cur_state % num_rows
    for av in range(4):
        s = act_dist[cur_act]['a'][av](cur_state, i, j)
        r = getReward(s)
        st_list.append((act_dist[cur_act]['p'][av], s, r))

    return st_list


def getAction(state, q, eps):
    max_act =  max(q[state])
    max_act_ct = np.count_nonzero(q[state] == max_act)
    prob = [((1-eps)/max_act_ct)+(eps/4) if av == max_act else eps/4 for av in q[state]]
    return random.choices(population=[0, 1, 2, 3], weights=prob)[0]

def getPolicy(q, eps):
    pi = []
    for state in range(num_rows*num_cols):
        max_act =  max(q[state])
        max_act_ct = np.count_nonzero(q[state] == max_act)
        pi.append([((1-eps)/max_act_ct)+(eps/4) if av == max_act else eps/4 for av in q[state]])
    return pi

def getReward(state):
    if state == 24:
        return 10
    elif state == 22:
        return -10
    return 0

def prioritySweep(eps, theta, max_iter):
    
    mse_list = []

    pq = []
    def priorityQueuePush(new_p, state):
      for idx, (old_p, s) in enumerate(pq):
          if s == state:
              if old_p <= new_p:
                  break
              pq[idx] = (new_p, state)
              heapq.heapify(pq)
              break
      else:
        heapq.heappush(pq, (new_p, state))

    q = np.zeros((num_rows*num_cols, 4), dtype="float")
    gamma = 0.9

    pred = {}
    for state in range(num_rows*num_cols):
        if state != 12 and state != 17 and state != 24:
            for action in [0, 1, 2, 3]:
                next_state = getState(state, action)
                if next_state in pred:
                    pred[next_state].add((state, action))
                else:
                    pred[next_state] = {(state, action)}
  
    while True:
        state = np.random.randint(num_rows*num_cols)
        while state == 12 or state == 17 or state == 24:
            state = np.random.randint(num_rows*num_cols)
        action = getAction(state, q, eps)
        next_state = getState(state, action)
        rew = getReward(next_state)
        q_old = q[state][action]
        q[state][action] = sum([prob*(rew + gamma*max(q[next_state])) for prob, next_state, rew in getAllStates(state, action)])
        p = q[state][action] - q_old
        if p > theta:
          if q_old == max(q[state]) or q[state][action] == max(q[state]):
            priorityQueuePush(-p, state)

        num_iter = 1
        while len(pq) > 0 and num_iter < max_iter:
          _, cur_state = heapq.heappop(pq)
          for pred_state, pred_act in pred[cur_state]:
              q_old = q[pred_state][pred_act]
              q[pred_state][pred_act] = sum([prob*(rew + gamma*max(q[next_state])) for prob, next_state, rew in getAllStates(pred_state, pred_act)])
              p = q[pred_state][pred_act] - q_old
              if p > theta:
                  num_iter += 1
                  if q_old == max(q[pred_state]) or q[pred_state][pred_act] == max(q[pred_state]):
                    priorityQueuePush(-p, pred_state)
        
        v = np.sum(np.array(getPolicy(q, eps)) * q, axis=1)
        mse = np.sum((v_op.flatten() - v)**2) / 25
        mse_list.append(mse)
        if mse < 0.5:
          break
        eps = max(0.1, eps-0.02)

    print(np.sum(np.array(getPolicy(q, eps)) * q, axis=1))
    print(len(mse_list))
    return mse_list, q

eps = 1
repeat = 20
mse_list = []
min_episodes = float("inf")
q_sum = np.zeros((num_rows*num_cols, 4), dtype="float")
for i in range(repeat):
    print(i)
    mse, q = prioritySweep(eps, 0.1, 100)
    mse_list.append(mse)
    if len(mse) < min_episodes:
        min_episodes = len(mse)
    q_sum += q

for i in range(repeat):
    mse_list[i] = mse_list[i][:min_episodes]
mse_list = np.sum(mse_list, axis = 0) / repeat

plt.plot(range(min_episodes), mse_list)
plt.xlabel("Number of Episodes")
plt.ylabel("Value Function - Mean Squared Error")
plt.savefig("ps_gridWorld")
plt.close()