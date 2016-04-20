import numpy as np
#Setting the seed before other dependent imports
np.random.seed(0)
import theano
import theano.tensor as T
import pandas as pd
import matplotlib.pyplot as plt
from layers import LSTMLayer, InputLayer, FullyConnectedLayer
from lib import get_params, make_caches, SGD, momentum
from basic_lstm import basicLSTM

def _load_data(data, n_prev=3):  
    """
    data should be pd.DataFrame()
    """
    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.data.iloc[i:i+n_prev])
        docY.append(data.response.iloc[i+n_prev]) 

    alsX = np.array(docX)
    alsY = np.array(docY)
    return np.matrix(alsX), np.matrix(alsY).T



def data_generation(stopTime, numPoints, n_prev):
    """
    Inputs:
    stopTime - stop time for the sine series
    numPoints - Resolution
    n_prev - previous points to include for prediction
    """
    if n_prev > 1:
        startTime = 0
        time_vec = np.linspace(startTime, stopTime, num=numPoints)
        x = np.sin(time_vec)
        data = pd.DataFrame({"data":x, "response":x})
        data.response = data.response.shift(-1)
        #Not selecting the last point since the response is a Nan
        data = data.iloc[:-1]
        data_mat, resp_mat = _load_data(data, n_prev=n_prev)
    else:
        time_vec = np.linspace(0, 20, numPoints)
        data = np.sin(time_vec)
        data_rolled = np.roll(data, -1)[:-1]
        data_mat = np.matrix(data[:-1]).T
        resp_mat = np.matrix(data_rolled).T

    return data_mat, resp_mat, time_vec[n_prev:]

def plot_predicted_series(lstm_model, data_mat, resp_mat, time_vec, it_num, n_prev):
    plt.plot(time_vec, lstm_model.predict(data_mat)[0], label='Predicted' )
    plt.plot(time_vec, resp_mat, label='Actual')
    plt.title('Predicted plots for size '+str(n_prev)+' and '+str(it_num)+\
            ' training iterations')
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.ylim([-2, 2])
    base_path = 'graphs/inconsistency-checks/predicted-graphs/'
    plt.savefig(base_path+'predicted_n_prev_'+str(n_prev)+'_iteration_'+str(it_num)+\
            '.png')
    plt.close()


def plot_error(error_df, n_prev, Training=True):
    if Training == True:
        str_val = 'Training'
    else:
        str_val = 'Test'
    plt.plot(error_df.index, error_df.Error)
    plt.title(str_val+' Error variation with Iterations')
    plt.xlabel('Iterations')
    plt.ylabel(str_val+' Error')
    base_url = 'graphs/inconsistency-checks/error-graphs/'
    plt.savefig(base_url+'error_n_prev_'+str(n_prev)+'_2.png')
    plt.close()
#n_prev = 1 represents one to one mapping of input and output
#n_prev = n represents n to one mapping of input and output
#Creating the data
stopTime = 20
numPoints = 200
n_prev = 1
n_iterations = 1000
data_mat, resp_mat, time_vec = data_generation(stopTime, numPoints, n_prev)
#Creating the LSTM model
lstm_model = basicLSTM(n_prev)
#eta - learning rate
eta = 0.5
#alpha - momentum hyperparameter
alpha = 1.0

error_list = []
for i in range(n_iterations):
    r_cost = lstm_model.train(data_mat, resp_mat, eta, alpha)
    error_list.append([i+1, float(r_cost[0])])
    if (i+1)<100 and (i+1)%10 == 0:
        print "iteration: %s, cost: %s" % (i+1, float(r_cost[0]))
        plot_predicted_series(lstm_model, data_mat, resp_mat, time_vec, i+1, n_prev)
    if (i+1)%100 == 0:
        print "iteration: %s, cost: %s" % (i+1, float(r_cost[0]))
        plot_predicted_series(lstm_model, data_mat, resp_mat, time_vec, i+1, n_prev)
error_df = pd.DataFrame(np.array(error_list), columns=['Iteration','Error'])
error_df = error_df.set_index('Iteration')
error_df.to_csv('results/inconsistency-checks/error_trajectory2.csv')
plot_error(error_df, n_prev)



