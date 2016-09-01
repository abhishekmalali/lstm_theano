import datagen as dg
from LSTM import basicLSTM

dgenObj = dg.GenFunction()
data_mat, resp_mat, time_vec = dgenObj.regular_sin(500, 2, 20, noise=0.2)
numSelPoints = 250
data_mat_tr = data_mat[:numSelPoints/2,:]
resp_mat_tr = resp_mat[:numSelPoints/2,:]
n_prev=1
alpha = 1.0
n_iterations = 5000

lstm_model = basicLSTM(1, num_cells=5)

error_list = []
for i in range(n_iterations):
    r_cost = lstm_model.train(data_mat_tr, resp_mat_tr, alpha)
    print i, r_cost
    error_list.append([i+1, float(r_cost[0])])

print error_list
