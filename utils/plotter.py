import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
sns.set_style("whitegrid")


class Plotter():

    def __init__(self):
        self.error_df = None
        return

    def plot_predicted_series(self, lstm_model, data_mat, resp_mat, time_vec,
                              it_num, base_path):
        plt.plot(time_vec, lstm_model.predict(data_mat)[0], marker='.',
                 label='Predicted')
        plt.plot(time_vec, resp_mat, marker='.', label='Actual')
        plt.title('Predicted plots after ' + str(it_num) +
                  ' training iterations')
        plt.xlabel('Time')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.savefig(base_path+'Iteration_' + str(it_num) + '.png')
        plt.close()

    def plot_error(self, base_path, file_name=None, Training=True):
        if Training is True:
            str_val = 'Training'
        else:
            str_val = 'Test'
        plt.plot(self.error_df.index, self.error_df.Error)
        plt.title(str_val+' Error variation with Iterations')
        plt.xlabel('Iterations')
        plt.ylabel(str_val+' Error')
        plt.savefig(base_path+'error_noise_' +
                    file_name+'.png')
        plt.close()

    def create_error_df(self, error_list, save_path=None, file_name=None):
        error_df = pd.DataFrame(np.array(error_list),
                                columns=['Iteration', 'Error'])
        error_df = error_df.set_index('Iteration')
        self.error_df = error_df
        if save_path is not None:
            error_df.to_csv(save_path + file_name + '.csv')
