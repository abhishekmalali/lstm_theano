import numpy as np
#Setting the seed before other dependent imports
np.random.seed(0)
import theano
import theano.tensor as T
import pandas as pd
import matplotlib.pyplot as plt
from layers import LSTMLayer, InputLayer, FullyConnectedLayer
from lib import get_params, make_caches, SGD, momentum, create_optimization_updates
from basic_lstm import basicLSTM
import os

def mv_data_generation_irregular(numPoints, numSelPoints, L, noiseSD):
    mean = np.zeros(numPoints)
    vec = np.array([[j for j in range(0,numPoints)] for k in range(0,numPoints)])
    covMat = np.exp(-(vec - vec.T)**2/float(L**2))
    noise = np.random.normal(loc=0, scale=noiseSD ,size=numPoints)
    data = np.random.multivariate_normal(mean, covMat, size=(1,))[0] + noise
    time = np.array(range(numPoints))
    index = np.sort(np.random.choice(range(numPoints), size=numSelPoints, replace=False))
    time_irr = time[index]
    data_irr = data[index]
    delta_t = [0] + list(np.array(time_irr[1:]) - np.array(time_irr[:-1]))
    derivative = np.divide(np.diff(data_irr), np.diff(time_irr))
    magDerivative = np.lib.pad(derivative, (1,0), 'constant', constant_values=(0,0))
    dDerivative = np.lib.pad(np.diff(magDerivative), (1,0), 'constant', constant_values=(0,0))
    data_mat = np.matrix([list(data_irr), delta_t, list(magDerivative), list(dDerivative)]).T
    data_mat = data_mat[:-1,:]
    resp_mat = np.matrix(np.roll(data_irr, -1)[:-1]).T
    time_vec = time_irr[1:]
    return data_mat, resp_mat, time_vec

def plot_predicted_series(lstm_model, data_mat, resp_mat, time_vec, it_num, n_prev, base_path):
    plt.plot(time_vec, lstm_model.predict(data_mat)[0], marker='.', label='Predicted' )
    plt.plot(time_vec, resp_mat, marker = '.',label='Actual')
    plt.title('Predicted plots for size '+str(n_prev)+' and '+str(it_num)+\
            ' training iterations')
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.legend() 
    plt.savefig(base_path+'predicted_n_prev_'+str(n_prev)+'_iteration_'+str(it_num)+\
            '.png')
    plt.close()

def plot_error(error_df, n_prev, noiseSD, base_path, Training=True):
    if Training == True:
        str_val = 'Training'
    else:
        str_val = 'Test'
    plt.plot(error_df.index, error_df.Error)
    plt.title(str_val+' Error variation with Iterations')
    plt.xlabel('Iterations')
    plt.ylabel(str_val+' Error')
    plt.savefig(base_path+'error_n_prev_'+str(n_prev)+'_noise_'+str(noiseSD)+'.png')
    plt.close()

graphs_path = os.path.join(os.getcwd(),'graphs')
store_path = os.path.join(graphs_path, 'mv-irregular-with-noise-test-ad')
pred_path = os.path.join(store_path, 'predicted-graphs')
err_path = os.path.join(store_path, 'error-graphs')
if not os.path.isdir(err_path):
    os.makedirs(err_path)
err_path = err_path + '/'

L = 20
numPoints = 200
numSelPoints = 60
noiseSD = 0.2
data_mat, resp_mat, time_vec = mv_data_generation_irregular(numPoints,\
        numSelPoints, L, noiseSD)
data_mat_tr = data_mat[:numSelPoints/2,:]
resp_mat_tr = resp_mat[:numSelPoints/2,:]
#alpha - momentum hyperparameter
n_prev=1
alpha = 1.0
n_iterations = 5000
numCellArray = np.arange(5, 30, 5)

for num in numCellArray:
    #Creating the folder for storing the folders
    numPath = os.path.join(pred_path, str(num)+'cell_states')
    if not os.path.isdir(numPath):
        os.makedirs(numPath)
    numPath = numPath + '/'
    lstm_model = basicLSTM(4, num_cells=num)
    error_list = []
    for i in range(n_iterations):
        r_cost = lstm_model.train(data_mat_tr, resp_mat_tr, alpha)
        error_list.append([i+1, float(r_cost[0])])
        if (i+1)<100 and (i+1)%10 == 0:
            print "iteration: %s, cost: %s" % (i+1, float(r_cost[0]))
            plot_predicted_series(lstm_model, data_mat, resp_mat,\
                    time_vec, i+1, n_prev, numPath)
        if (i+1)%100 == 0 and i<1000:
            print "iteration: %s, cost: %s" % (i+1, float(r_cost[0]))
            plot_predicted_series(lstm_model, data_mat, resp_mat,\
                    time_vec, i+1, n_prev, numPath)
        if (i+1)%1000 == 0 and i > 1000:
            print "iteration: %s, cost: %s" % (i+1, float(r_cost[0]))
            plot_predicted_series(lstm_model, data_mat, resp_mat,\
                    time_vec, i+1, n_prev, numPath)
    error_df = pd.DataFrame(np.array(error_list), columns=['Iteration','Error'])
    error_df = error_df.set_index('Iteration')
    error_df.to_csv(err_path+'error_trajectory_numcells_'+str(num)+'.csv')
    plot_error(error_df, n_prev, num, err_path)

