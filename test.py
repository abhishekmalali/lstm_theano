import datagen as dg
from LSTM import basicLSTM
from utils import *
import numpy as np

plotObj = Plotter()
dgenObj = dg.GenFunction()
data_mat, resp_mat, time_vec = dgenObj.regular_sin(500, 2, 20, noise=0.2)
numSelPoints = 250
data_mat_tr = data_mat[:numSelPoints/2, :]
resp_mat_tr = resp_mat[:numSelPoints/2, :]
alpha = 1.0
n_iterations = 5000


# Path manipulation
datahandler = DataHandler()
pred_path, err_path = datahandler.create_paths('graphs',
                                               'regular-with-noise-0.2')

numCellArray = np.arange(5, 30, 5)
error_list = []
for num in numCellArray:
    lstm_model = basicLSTM(1, num_cells=num)
    graph_path = datahandler.create_new_pred_path(str(num))
    for i in range(n_iterations):
        r_cost = lstm_model.train(data_mat_tr, resp_mat_tr, alpha)
        error_list.append([i+1, float(r_cost[0])])
        if i % 100 == 0:
            plotObj.plot_predicted_series(lstm_model, data_mat, resp_mat,
                                          time_vec, i,  graph_path)

    plotObj.create_error_df(error_list, save_path=err_path,
                            file_name='regular-with-noise-0.2-'+str(num))
    plotObj.plot_error(err_path, 'regular-with-noise-0.2-'+str(num))
