import numpy as np
import matplotlib.pyplot as plt


bel = np.hstack((np.zeros(9), 1, np.zeros(10)))
print(bel)
N = bel.size 

def discrete_bayes_filter(bel, u):
  '''
    u: if True ->, if False <-
  '''
  bel_new = np.zeros(len(bel))
  P_i = 0.25    # P(x_i | ->, x_i)
  P_i_1 = 0.5   # P(x_i+1 | ->, x_i)
  P_i_2 = 0.25  # P(x_i+2 | ->, x_i)
  P_border = 1  # P(x_N | ->, x_N) or  P(x_0 | <-, x_0)
  P_N_1 = 0.25  # P(x_N-1 | ->, x_N-1)
  P_1 = 0.25    # P(x_1 | <-, x_1)
  P_N = 0.75    # P(x_N | ->, x_N-1)
  P_0 = 0.75    # P(x_0 | <-, x_1)

  if u:
    for i in range(len(bel)):
      if i == 0:
        bel_new[i] = P_i * bel[i]
      elif i == 1:
        bel_new[i] = P_i * bel[i] + P_i_1 * bel[i-1]
      elif i == len(bel)-2:
        bel_new[i] = P_N_1 * bel[i] + P_i_1 * bel[i-1] + P_i_2 * bel[i-2]
      elif i == len(bel)-1:
        bel_new[i] = P_N * bel[i-1] + P_border * bel[i] + P_i_2 * bel[i-2]
      else:
        bel_new[i] = P_i * bel[i] + P_i_1 * bel[i-1] + P_i_2 * bel[i-2]
  
  else:
    for i in range(len(bel)):
      if i == 0:
        bel_new[i] = P_0 * bel[i+1] + P_border * bel[i] + P_i_2 * bel[i+2]
      elif i == 1:
        bel_new[i] = P_1 * bel[i] + P_i_1 * bel[i+1] + P_i_2 * bel[i+2]
      elif i == len(bel)-2:
        bel_new[i] = P_i * bel[i] + P_i_1 * bel[i+1]
      elif i == len(bel)-1:
        bel_new[i] = P_i * bel[i]
      else:
        bel_new[i] = P_i * bel[i] + P_i_1 * bel[i+1] + P_i_2 * bel[i+2]
      
  return bel_new


for i in range(9):
  bel = discrete_bayes_filter(bel, True)
  print(np.sum(bel))
for i in range(3):
  bel = discrete_bayes_filter(bel, False)
  print(np.sum(bel))

print(bel)
print(bel.max())
print(f"robot is in cell {np.argmax(bel) + 1}")